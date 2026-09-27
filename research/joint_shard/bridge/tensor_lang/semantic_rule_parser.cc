#include <cctype>
#include <charconv>
#include <stdexcept>
#include <utility>

#include "research/joint_shard/bridge/tensor_lang/semantic_rule_internal.h"

namespace joint_shard {
namespace semantic_detail {
namespace {
class Parser {
 public:
  Parser(std::string_view source, std::string name)
      : source_(source), name_(std::move(name)) {
    next();
  }

  std::vector<Rule> parse() {
    std::vector<Rule> rules;
    while (token_.kind != Kind::End) {
      expect("rule");
      Rule rule;
      rule.line = token_.line;
      rule.name = expect(Kind::Word).text;
      expect("{");
      rule.lhs = term();
      expect("=>");
      rule.rhs = term();
      expect("where");
      do {
        Predicate predicate;
        predicate.name = expect(Kind::Word).text;
        expect("(");
        predicate.subject =
            expect(token_.kind == Kind::Variable ? Kind::Variable : Kind::Word)
                .text;
        if (token_.text == ",") {
          next();
          const Token index = expect(Kind::Number);
          unsigned operand = 0;
          const auto [end, error] =
              std::from_chars(index.text.data(),
                              index.text.data() + index.text.size(), operand);
          if (error != std::errc{} ||
              end != index.text.data() + index.text.size())
            fail("invalid operand index");
          predicate.operand = operand;
        }
        expect(")");
        rule.predicates.push_back(std::move(predicate));
        if (token_.text != "and") break;
        next();
      } while (true);
      expect(";");
      expect("}");
      validation_line_ = rule.line;
      validate(rule);
      if (!names_.insert(rule.name).second)
        fail("duplicate rule name " + rule.name);
      validation_line_.reset();
      rules.push_back(std::move(rule));
    }
    if (rules.empty()) fail("expected at least one rule");
    return rules;
  }

 private:
  enum class Kind { Word, Variable, Number, Symbol, End };
  struct Token {
    Kind kind = Kind::End;
    std::string text;
    std::size_t line = 1;
    std::size_t column = 1;
  };

  [[noreturn]] void fail(const std::string& message) const {
    throw std::invalid_argument(
        name_ + ":" + std::to_string(validation_line_.value_or(token_.line)) +
        ":" + std::to_string(validation_line_ ? 1 : token_.column) + ": " +
        message);
  }
  void next() {
    while (position_ < source_.size()) {
      if (source_[position_] == '#') {
        while (position_ < source_.size() && source_[position_] != '\n') {
          ++position_;
          ++column_;
        }
        continue;
      }
      if (!std::isspace(static_cast<unsigned char>(source_[position_]))) break;
      if (source_[position_] == '\n') {
        ++line_;
        column_ = 1;
      } else {
        ++column_;
      }
      ++position_;
    }
    token_ = {Kind::End, {}, line_, column_};
    if (position_ == source_.size()) return;
    const std::size_t start = position_;
    const std::size_t start_column = column_;
    const char ch = source_[position_++];
    ++column_;
    if (ch == '=' && position_ < source_.size() && source_[position_] == '>') {
      ++position_;
      ++column_;
      token_ = {Kind::Symbol, "=>", line_, start_column};
    } else if (ch == '?') {
      while (position_ < source_.size() &&
             (std::isalnum(static_cast<unsigned char>(source_[position_])) ||
              source_[position_] == '_')) {
        ++position_;
        ++column_;
      }
      token_ = {Kind::Variable,
                std::string(source_.substr(start, position_ - start)), line_,
                start_column};
    } else if (std::isalpha(static_cast<unsigned char>(ch)) || ch == '_') {
      while (position_ < source_.size() &&
             (std::isalnum(static_cast<unsigned char>(source_[position_])) ||
              source_[position_] == '_' || source_[position_] == '.' ||
              source_[position_] == '-')) {
        ++position_;
        ++column_;
      }
      token_ = {Kind::Word,
                std::string(source_.substr(start, position_ - start)), line_,
                start_column};
    } else if (std::isdigit(static_cast<unsigned char>(ch))) {
      while (position_ < source_.size() &&
             std::isdigit(static_cast<unsigned char>(source_[position_]))) {
        ++position_;
        ++column_;
      }
      token_ = {Kind::Number,
                std::string(source_.substr(start, position_ - start)), line_,
                start_column};
    } else if (std::string_view("{}(),;").find(ch) != std::string_view::npos) {
      token_ = {Kind::Symbol, std::string(1, ch), line_, start_column};
    } else {
      fail(std::string("unexpected character '") + ch + "'");
    }
  }
  Token expect(Kind kind) {
    if (token_.kind != kind) fail("unexpected token '" + token_.text + "'");
    Token result = token_;
    next();
    return result;
  }
  void expect(std::string_view spelling) {
    if (token_.text != spelling)
      fail("expected '" + std::string(spelling) + "', got '" + token_.text +
           "'");
    next();
  }
  Term term() {
    if (token_.kind == Kind::Variable) {
      auto name = expect(Kind::Variable).text;
      if (name.size() == 1) fail("empty tensor variable");
      return {std::move(name), true, {}};
    }
    const std::string name = expect(Kind::Word).text;
    expect("(");
    std::vector<Term> children;
    if (token_.text != ")") {
      do {
        children.push_back(term());
        if (token_.text != ",") break;
        next();
      } while (true);
    }
    expect(")");
    return {name, false, std::move(children)};
  }
  static bool opKind(const std::string& name, OpKind& result) {
    auto* schema = lookupOpSchema(name);
    if (!schema) return false;
    result = schema->op;
    return true;
  }

  static void collect(const Term& term, std::unordered_set<std::string>& vars,
                      std::unordered_set<std::string>& ops) {
    if (term.variable) {
      vars.insert(term.name);
      return;
    }
    OpKind kind;
    if (term.name != "scale" && !opKind(term.name, kind)) ops.insert(term.name);
    for (const auto& child : term.children) collect(child, vars, ops);
  }
  void validate(const Rule& rule) const {
    std::unordered_set<std::string> lhs_vars, lhs_ops, rhs_vars, rhs_ops;
    collect(rule.lhs, lhs_vars, lhs_ops);
    collect(rule.rhs, rhs_vars, rhs_ops);
    std::unordered_map<std::string, std::size_t> arities;
    const auto check = [&](auto&& self, const Term& term) -> void {
      if (term.variable) return;
      if (term.name == "scale") {
        if (term.children.size() != 2) fail("scale requires two operands");
      } else if (auto* schema = lookupOpSchema(term.name)) {
        if (!schema->attribute_free)
          fail("concrete operator needs attributes: " + term.name);
        if (schema->arity != term.children.size())
          fail("wrong arity for " + term.name);
      } else {
        if (!std::isupper(static_cast<unsigned char>(term.name[0])))
          fail("unknown concrete operator " + term.name +
               "; operator variables start uppercase");
        auto [it, inserted] = arities.emplace(term.name, term.children.size());
        if (!inserted && it->second != term.children.size())
          fail("inconsistent arity for " + term.name);
      }
      for (const auto& child : term.children) self(self, child);
    };
    check(check, rule.lhs);
    check(check, rule.rhs);
    if (rule.lhs.variable) fail("left side must have an operator root");
    for (const auto& predicate : rule.predicates) {
      if (predicate.name == "Scalar" || predicate.name == "Uniform") {
        if (predicate.operand || !lhs_vars.contains(predicate.subject))
          fail(
              "value predicate requires one tensor variable bound on the left");
        continue;
      }
      const bool indexed =
          predicate.name == "LinearIn" || predicate.name == "HomogeneousIn";
      if (predicate.name != "Elementwise" && predicate.name != "Commutative" &&
          predicate.name != "Associative" && predicate.name != "Involution" &&
          !indexed)
        fail("unknown operator property " + predicate.name);
      if (indexed != predicate.operand.has_value())
        fail("wrong property operand index syntax");
      if (!lhs_ops.contains(predicate.subject))
        fail("property operator must be bound by the left side");
      if (predicate.operand &&
          *predicate.operand >= arities.at(predicate.subject))
        fail("property operand is outside the operator arity");
    }
    for (const auto& name : rhs_vars)
      if (!lhs_vars.contains(name))
        fail("right side has unbound variable " + name);
    for (const auto& name : rhs_ops)
      if (!lhs_ops.contains(name))
        fail("right side has unbound operator " + name);
    if (lhs_vars.empty())
      fail("left side must bind at least one tensor variable");
  }

  std::string_view source_;
  std::string name_;
  std::optional<std::size_t> validation_line_;
  std::unordered_set<std::string> names_;
  std::size_t position_ = 0;
  std::size_t line_ = 1;
  std::size_t column_ = 1;
  Token token_;
};

}  // namespace
std::vector<Rule> parseRules(std::string_view source, std::string name) {
  return Parser(source, std::move(name)).parse();
}
}  // namespace semantic_detail

}  // namespace joint_shard

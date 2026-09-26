#include <algorithm>
#include <cctype>
#include <charconv>
#include <fstream>
#include <functional>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "llvm/Support/Casting.h"
#include "research/joint_shard/bridge/tensor_lang/tensor_rewrites.h"

namespace {
struct Term {
  std::string name;
  bool variable = false;
  std::vector<Term> children;
};
struct Predicate {
  std::string name;
  std::string subject;
  std::optional<unsigned> operand;
};
struct Rule {
  std::string name;
  Term lhs;
  Term rhs;
  std::vector<Predicate> predicates;
  std::size_t line = 1;
};

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

using TensorBindings = std::unordered_map<std::string, eggc::Id>;
struct OperatorOccurrence {
  TensorNode node;
  eggc::Id eclass;
};
struct Bindings {
  TensorBindings tensors;
  std::unordered_map<std::string, std::vector<OperatorOccurrence>> operators;
};
struct SearchState {
  const eggc::StopCheck& stop;
  const SemanticRuleOptions& options;
  SemanticRuleStats& stats;
  NumericalPolicy policy;
  std::size_t visits = 0;
  std::size_t matches = 0;
  bool budget = false;
  bool cancelled() {
    if (stop && stop()) return true;
    if (visits >= options.visit_limit || matches >= options.match_limit) {
      if (!budget) ++stats.budget_stops;
      budget = true;
      return true;
    }
    return false;
  }
};
// Follow only broadcasts: this keeps scalar identity explicit and does not
// invent a scalar from a tensor splat. All helper searches share rule budgets.
using ScalarSink = std::function<bool(eggc::Id)>;
bool findScalars(eggc::Id id, const TensorEGraph& graph, SearchState& state,
                 std::unordered_set<eggc::Id>& seen, const ScalarSink& sink) {
  if (state.cancelled()) return false;
  id = graph.find(id);
  if (!seen.insert(id).second) return true;
  const auto& facts = tensorFacts(graph, id);
  if (facts.type && facts.type.getRank() == 0) return sink(id);
  for (const auto& node : graph.nodes(id)) {
    if (state.cancelled()) return false;
    ++state.visits;
    ++state.stats.node_visits;
    if (node.op == OpKind::BroadcastInDim &&
        !findScalars(node.operands[0], graph, state, seen, sink))
      return false;
  }
  return true;
}
using BindingSink = std::function<bool(const Bindings&)>;
// Depth-first streaming matching; no intermediate Cartesian-product lists.
bool matchTerm(const Term& term, eggc::Id id, const TensorEGraph& graph,
               const Bindings& initial, SearchState& state,
               const BindingSink& sink) {
  if (state.cancelled()) return false;
  id = graph.find(id);
  if (term.variable) {
    Bindings next = initial;
    auto [found, inserted] = next.tensors.emplace(term.name, id);
    if (!inserted && graph.find(found->second) != id) return true;
    return sink(next);
  }
  if (term.name == "scale") {
    for (const auto& node : graph.nodes(id)) {
      if (state.cancelled()) return false;
      ++state.visits;
      ++state.stats.node_visits;
      if (node.op != OpKind::Multiply) continue;
      // The RHS builder puts the scalar first. Matching a scalar on the right
      // therefore also needs permission to commute this multiply (e.g. strict
      // floating evaluation cannot freely reorder NaN operands).
      for (unsigned slot = 0; slot < 2; ++slot) {
        if (slot == 1) {
          std::vector<TensorFacts> operands{
              tensorFacts(graph, node.operands[0]),
              tensorFacts(graph, node.operands[1])};
          if (!queryProperty(node, Commutative{},
                             {operands, tensorFacts(graph, id), state.policy,
                              state.options.permissions})
                   .allowed)
            continue;
        }
        std::unordered_set<eggc::Id> seen;
        if (!findScalars(
                node.operands[slot], graph, state, seen, [&](eggc::Id scalar) {
                  return matchTerm(term.children[0], scalar, graph, initial,
                                   state, [&](const Bindings& bound) {
                                     return matchTerm(term.children[1],
                                                      node.operands[1 - slot],
                                                      graph, bound, state,
                                                      sink);
                                   });
                }))
          return false;
      }
    }
    return true;
  }
  const auto* schema = lookupOpSchema(term.name);
  for (const auto& node : graph.nodes(id)) {
    if (state.cancelled()) return false;
    ++state.visits;
    ++state.stats.node_visits;
    if (node.operands.size() != term.children.size()) continue;
    if (schema &&
        (node.op != schema->op || !std::holds_alternative<NoAttrs>(node.attrs)))
      continue;
    Bindings bound = initial;
    if (!schema) {
      auto& occurrences = bound.operators[term.name];
      if (!occurrences.empty() && !occurrences.front().node.matches(node))
        continue;
      occurrences.push_back({node, id});
    }
    const auto children = [&](auto&& self, size_t index,
                              const Bindings& binding) -> bool {
      if (state.cancelled()) return false;
      if (index == term.children.size()) return sink(binding);
      return matchTerm(
          term.children[index], node.operands[index], graph, binding, state,
          [&](const Bindings& next) { return self(self, index + 1, next); });
    };
    if (!children(children, 0, bound)) return false;
  }
  return true;
}
OpProperty toProperty(const Predicate& predicate) {
  if (predicate.name == "Elementwise") return Elementwise{};
  if (predicate.name == "Commutative") return Commutative{};
  if (predicate.name == "Associative") return Associative{};
  if (predicate.name == "Involution") return Involution{};
  if (predicate.name == "LinearIn") return LinearIn{*predicate.operand};
  return HomogeneousIn{*predicate.operand};
}
PropertyDecision checkNode(const TensorNode& node, TensorFacts result,
                           const std::vector<TensorFacts>& operands,
                           const Predicate& predicate, NumericalPolicy policy,
                           std::optional<NumericalPermissions> permissions) {
  return queryProperty(node, toProperty(predicate),
                       {operands, result, policy, permissions});
}
bool isUniformTensor(const TensorEGraph& graph, eggc::Id id,
                     std::unordered_set<eggc::Id>& seen, SearchState& state) {
  if (state.cancelled()) return false;
  id = graph.find(id);
  if (!seen.insert(id).second) return false;
  const auto& facts = tensorFacts(graph, id);
  if (facts.type && facts.type.getRank() == 0) return true;
  if (auto value =
          llvm::dyn_cast_or_null<mlir::DenseElementsAttr>(facts.constant))
    if (value.isSplat()) return true;
  for (const auto& node : graph.nodes(id)) {
    if (state.cancelled()) return false;
    ++state.visits;
    ++state.stats.node_visits;
    if (node.op == OpKind::BroadcastInDim &&
        isUniformTensor(graph, node.operands[0], seen, state))
      return true;
  }
  return false;
}
struct Prepared {
  std::optional<eggc::Id> reference;
  TensorNode node{};
  TensorFacts facts;
  std::vector<Prepared> children;
};
std::optional<Prepared> prepare(const Term& term, const Bindings& bindings,
                                const TensorEGraph& graph, const Rule& rule,
                                NumericalPolicy policy,
                                const SemanticRuleOptions& options,
                                std::string& rejection) {
  Prepared result;
  if (term.variable) {
    result.reference = graph.find(bindings.tensors.at(term.name));
    result.facts = tensorFacts(graph, *result.reference);
    return result;
  }
  std::vector<TensorFacts> facts;
  for (const auto& child : term.children) {
    auto prepared =
        prepare(child, bindings, graph, rule, policy, options, rejection);
    if (!prepared) return std::nullopt;
    facts.push_back(prepared->facts);
    result.children.push_back(std::move(*prepared));
  }
  if (term.name == "scale") {
    const auto scalar_type = facts[0].type;
    const auto target_type = facts[1].type;
    if (!scalar_type || scalar_type.getRank() != 0 || !target_type ||
        scalar_type.getElementType() != target_type.getElementType()) {
      rejection = "RHS scale requires a rank-zero scalar of the tensor dtype";
      return std::nullopt;
    }
    if (target_type.getRank() != 0) {
      Prepared broadcast;
      broadcast.node = {
          OpKind::BroadcastInDim, BroadcastAttrs{{}, target_type}, {0}};
      auto inference = inferTensorNode(
          broadcast.node, std::span<const TensorFacts>(facts.data(), 1));
      if (!inference.valid()) {
        rejection = "RHS scale broadcast: " + inference.reason;
        return std::nullopt;
      }
      broadcast.facts = inference.facts;
      broadcast.children.push_back(std::move(result.children[0]));
      result.children[0] = std::move(broadcast);
      facts[0] = result.children[0].facts;
    }
    result.node = {OpKind::Multiply, NoAttrs{}, {0, 0}};
    auto inference = inferTensorNode(result.node, facts);
    if (!inference.valid()) {
      rejection = "RHS scale: " + inference.reason;
      return std::nullopt;
    }
    result.facts = inference.facts;
    return result;
  }
  auto* schema = lookupOpSchema(term.name);
  result.node = schema ? TensorNode{schema->op, NoAttrs{}, {}}
                       : bindings.operators.at(term.name).front().node;
  // Only arity is relevant to standalone inference; local IDs are placeholders.
  result.node.operands.assign(facts.size(), 0);
  auto inference = inferTensorNode(result.node, facts);
  if (!inference.valid()) {
    rejection = "RHS: " + inference.reason;
    return std::nullopt;
  }
  result.facts = inference.facts;
  if (!schema)
    for (const auto& predicate : rule.predicates)
      if (predicate.subject == term.name) {
        auto decision = checkNode(result.node, result.facts, facts, predicate,
                                  policy, options.permissions);
        if (!decision.allowed) {
          rejection = "RHS: " + decision.reason;
          return std::nullopt;
        }
      }
  return result;
}
eggc::Id instantiate(const Prepared& prepared, TensorEGraph& graph) {
  if (prepared.reference) return graph.find(*prepared.reference);
  auto node = prepared.node;
  node.operands.clear();
  for (const auto& child : prepared.children)
    node.operands.push_back(instantiate(child, graph));
  return graph.add(std::move(node));
}
std::string bindingKey(eggc::Id root, const Bindings& bindings) {
  std::vector<std::string> entries;
  for (const auto& [name, id] : bindings.tensors)
    entries.push_back(name + "=" + std::to_string(id));
  for (const auto& [name, occurrences] : bindings.operators)
    for (const auto& occurrence : occurrences)
      entries.push_back(name + "=" + std::to_string(occurrence.eclass) + ":" +
                        occurrence.node.format());
  std::sort(entries.begin(), entries.end());
  std::string key = std::to_string(root);
  for (const auto& entry : entries)
    key += "|" + std::to_string(entry.size()) + ":" + entry;
  return key;
}
TensorRewrite lower(Rule rule, NumericalPolicy policy,
                    SemanticRuleOptions options) {
  using Rewrite = TensorRewrite;
  const auto name = rule.name;
  return Rewrite{
      name, [rule = std::move(rule), policy, options](
                const TensorEGraph& graph, const Rewrite::Sink& sink,
                const eggc::StopCheck& stop) {
        SemanticRuleStats local;
        auto& stats = options.report ? (*options.report)[rule.name] : local;
        SearchState state{stop, options, stats, policy};
        std::unordered_set<std::string> seen;
        const auto* schema = lookupOpSchema(rule.lhs.name);
        std::vector<eggc::Id> candidates;
        if (rule.lhs.name == "scale")
          candidates = graph.classes_for_op(OpKind::Multiply);
        else if (schema)
          candidates = graph.classes_for_op(schema->op);
        else {
          // A property on the root operator restricts the discriminant index.
          auto predicate = std::find_if(
              rule.predicates.begin(), rule.predicates.end(),
              [&](const Predicate& p) { return p.subject == rule.lhs.name; });
          if (predicate == rule.predicates.end())
            candidates = graph.classes();
          else {
            std::unordered_set<eggc::Id> roots;
            for (int value = 0; value <= static_cast<int>(OpKind::Transpose);
                 ++value) {
              auto op = static_cast<OpKind>(value);
              auto properties = declaredProperties(op);
              if (std::find(properties.begin(), properties.end(),
                            toProperty(*predicate)) == properties.end())
                continue;
              for (auto id : graph.classes_for_op(op)) roots.insert(id);
            }
            candidates.assign(roots.begin(), roots.end());
            std::sort(candidates.begin(), candidates.end());
          }
        }
        for (auto root : candidates) {
          bool complete = matchTerm(
              rule.lhs, root, graph, {}, state, [&](const Bindings& binding) {
                if (state.cancelled()) return false;
                const auto reject = [&](std::string reason) {
                  ++stats.rejections[std::move(reason)];
                  return sink({root, {}, false});
                };
                if (!seen.insert(bindingKey(root, binding)).second) {
                  ++stats.duplicates;
                  return true;
                }
                ++state.matches;
                ++stats.structural_matches;
                for (const auto& predicate : rule.predicates) {
                  if (predicate.name == "Scalar" ||
                      predicate.name == "Uniform") {
                    auto id = binding.tensors.at(predicate.subject);
                    const auto type = tensorFacts(graph, id).type;
                    std::unordered_set<eggc::Id> visited;
                    bool allowed =
                        predicate.name == "Scalar"
                            ? type && type.getRank() == 0
                            : isUniformTensor(graph, id, visited, state);
                    // A cancelled helper must halt the search, not emit a
                    // semantic rejection or partial applications.
                    if ((stop && stop()) || state.budget) return false;
                    if (!allowed)
                      return reject(predicate.name + " requires proven value " +
                                    predicate.subject);
                    continue;
                  }
                  for (const auto& occurrence :
                       binding.operators.at(predicate.subject)) {
                    std::vector<TensorFacts> operands;
                    for (auto child : occurrence.node.operands)
                      operands.push_back(tensorFacts(graph, child));
                    auto decision = checkNode(
                        occurrence.node, tensorFacts(graph, occurrence.eclass),
                        operands, predicate, policy, options.permissions);
                    if (!decision.allowed) {
                      return reject(decision.reason);
                    }
                  }
                }
                std::string rejection;
                auto prepared = prepare(rule.rhs, binding, graph, rule, policy,
                                        options, rejection);
                if (!prepared ||
                    prepared->facts.type != tensorFacts(graph, root).type) {
                  return reject(
                      rejection.empty()
                          ? "RHS result type differs from matched root"
                          : rejection);
                }
                ++stats.accepted;
                return sink(
                    {root,
                     [prepared = std::move(*prepared), report = options.report,
                      name = rule.name](TensorEGraph& mutable_graph)
                         -> std::optional<eggc::Id> {
                       // Recheck the entire prepared tree against current facts
                       // before any add.
                       const auto validate =
                           [&](auto&& self,
                               const Prepared& p) -> InferenceResult {
                         if (p.reference)
                           return {InferenceStatus::Valid,
                                   tensorFacts(mutable_graph, *p.reference),
                                   {}};
                         std::vector<TensorFacts> operands;
                         for (const auto& child : p.children) {
                           auto result = self(self, child);
                           if (!result.valid()) return result;
                           operands.push_back(result.facts);
                         }
                         return inferTensorNode(p.node, operands);
                       };
                       if (!validate(validate, prepared).valid())
                         return std::nullopt;
                       if (report) ++(*report)[name].applied;
                       return instantiate(prepared, mutable_graph);
                     },
                     true});
              });
          if (!complete) return false;
        }
        return true;
      }};
}
}  // namespace
std::vector<TensorRewrite> parseSemanticRules(std::string_view source,
                                              NumericalPolicy policy,
                                              SemanticRuleOptions options) {
  Parser parser(source, options.source_name);
  auto parsed = parser.parse();
  std::vector<TensorRewrite> result;
  for (auto& rule : parsed) {
    bool enabled = true;
    for (const auto& predicate : rule.predicates) {
      if (predicate.name == "Associative" && !options.enable_associativity)
        enabled = false;
      if (predicate.name == "LinearIn" && !options.enable_linearity)
        enabled = false;
      if (predicate.name == "HomogeneousIn" && !options.enable_homogeneity)
        enabled = false;
    }
    if (enabled) result.push_back(lower(std::move(rule), policy, options));
  }
  return result;
}
std::vector<TensorRewrite> loadSemanticRulesFile(std::string_view path,
                                                 NumericalPolicy policy,
                                                 SemanticRuleOptions options) {
  std::ifstream input{std::string(path)};
  if (!input)
    throw std::runtime_error("cannot open semantic rules file: " +
                             std::string(path));
  std::ostringstream contents;
  contents << input.rdbuf();
  if (!input.good() && !input.eof())
    throw std::runtime_error("failed reading semantic rules file: " +
                             std::string(path));
  options.source_name = std::string(path);
  return parseSemanticRules(contents.str(), policy, std::move(options));
}

std::string_view defaultSemanticRules() {
  static constexpr std::string_view rules =
#include "research/joint_shard/bridge/tensor_lang/tensor_rules.inc"
      ;
  return rules;
}

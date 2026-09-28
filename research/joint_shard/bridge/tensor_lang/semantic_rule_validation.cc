#include <stdexcept>

#include "research/joint_shard/bridge/tensor_lang/semantic_rule_internal.h"

namespace joint_shard::semantic_detail {
namespace {
struct Variable {
  enum Kind { Tensor, Operator, Attribute } kind;
  unsigned detail;  // Operator arity or attribute value variant index.
  bool operator==(const Variable&) const = default;
};
unsigned attributeType(AttributeField field) {
  switch (field) {
    case AttributeField::Axes:
    case AttributeField::Permutation:
      return 0;
    case AttributeField::ReduceKind:
      return 1;
    case AttributeField::PrecisionConfig:
      return 2;
    case AttributeField::DotDimensions:
      return 3;
    case AttributeField::Broadcast:
      return 4;
  }
  throw std::logic_error("invalid attribute field");
}
}  // namespace
void validateRule(const Rule& rule) {
  const auto fail = [&](const std::string& reason) {
    throw std::invalid_argument("rule '" + rule.name + "': " + reason);
  };
  if (rule.name.empty()) fail("empty name");
  if (rule.lhs.kind == TermKind::TensorVariable)
    fail("LHS must be an operation");
  if (bool(rule.rhs) == bool(rule.builder))
    fail("provide exactly one RHS pattern or callback");
  std::unordered_map<std::string, Variable> variables;
  const auto bind = [&](const std::string& name, Variable variable, bool lhs) {
    if (name.empty()) fail("empty variable name");
    auto found = variables.find(name);
    if (found == variables.end()) {
      if (!lhs) fail("unbound RHS variable '" + name + "'");
      variables.emplace(name, variable);
    } else if (found->second != variable) {
      fail("inconsistent type or arity for variable '" + name + "'");
    }
  };
  const auto visit = [&](auto&& self, const Term& term, bool lhs) -> void {
    if (term.kind == TermKind::TensorVariable)
      bind(term.name, {Variable::Tensor, 0}, lhs);
    else if (term.kind == TermKind::OperatorVariable)
      bind(term.name,
           {Variable::Operator, static_cast<unsigned>(term.children.size())},
           lhs);
    for (const auto& field : term.attributes)
      if (const auto* name = std::get_if<std::string>(&field.expression))
        bind(*name, {Variable::Attribute, attributeType(field.field)}, lhs);
    for (const auto& child : term.children) self(self, child, lhs);
  };
  visit(visit, rule.lhs, true);
  if (rule.rhs) visit(visit, *rule.rhs, false);
  for (const auto& predicate : rule.predicates) {
    if (predicate.isRuleGuard()) continue;
    auto found = variables.find(predicate.subject);
    if (found == variables.end())
      fail("unbound guard variable '" + predicate.subject + "'");
    auto expected =
        predicate.isProperty() ? Variable::Operator : Variable::Tensor;
    if (found->second.kind != expected)
      fail("guard variable has wrong type: '" + predicate.subject + "'");
    if (predicate.isProperty() && predicate.operand &&
        *predicate.operand >= found->second.detail)
      fail("property operand exceeds matched operator arity");
  }
}
}  // namespace joint_shard::semantic_detail

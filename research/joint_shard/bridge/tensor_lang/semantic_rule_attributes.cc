#include "research/joint_shard/bridge/tensor_lang/semantic_rule_internal.h"

namespace joint_shard::semantic_detail {
bool matchAttributes(const Term& term, const TensorNode& node,
                     Bindings& bindings) {
  if (!validNodeSchema(node)) return false;
  for (const auto& constraint : term.attributes) {
    const auto& expression = constraint.expression;
    if (std::holds_alternative<patterns::Wildcard>(expression) ||
        std::holds_alternative<patterns::Inferred>(expression))
      continue;
    auto value = constraint.read(node.attrs);
    if (auto* variable = std::get_if<std::string>(&expression)) {
      auto [it, inserted] = bindings.attributes.emplace(*variable, value);
      if (!inserted && it->second != value) return false;
    } else if (auto* literal = std::get_if<AttributeValue>(&expression)) {
      // Empty dictionaries and absent optional metadata are equivalent for
      // literal matching. Node identity still retains the original handle.
      auto* expected = literal->get<mlir::DictionaryAttr>();
      auto* actual = value.get<mlir::DictionaryAttr>();
      if (expected && actual && (!*expected || expected->empty()) &&
          (!*actual || actual->empty()))
        continue;
      if (*literal != value) return false;
    } else
      return false;
  }
  return true;
}
std::optional<TensorNode> buildConcreteNode(
    const Term& term, const Bindings& bindings,
    std::span<const TensorFacts> operands, std::string& rejection) {
  TensorNode node{term.op, term.initial_attributes,
                  std::vector<eggc::Id>(operands.size(), 0)};
  for (const auto& constraint : term.attributes) {
    const auto& expression = constraint.expression;
    if (std::holds_alternative<patterns::Inferred>(expression)) continue;
    if (auto* variable = std::get_if<std::string>(&expression))
      constraint.write(node.attrs, bindings.attributes.at(*variable));
    else if (auto* literal = std::get_if<AttributeValue>(&expression))
      constraint.write(node.attrs, *literal);
    else {
      rejection = "RHS attribute requires a value or capture: " +
                  std::string(constraint.field);
      return std::nullopt;
    }
  }
  auto inferred = completeTensorNode(node, operands);
  if (!inferred.valid()) {
    rejection = "RHS: " + inferred.reason;
    return std::nullopt;
  }
  return node;
}
std::string formatAttribute(const AttributeValue& value) {
  return value.format();
}
}  // namespace joint_shard::semantic_detail

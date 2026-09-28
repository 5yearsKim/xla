#include "research/joint_shard/bridge/tensor_lang/tensorlang.h"

#include <stdexcept>
#include <type_traits>

#include "research/joint_shard/bridge/tensor_lang/tensor_attributes.h"

namespace joint_shard {

std::string TensorNode::format() const {
  return std::string(stableHloName(op)) +
         std::visit(
             [](const auto& attrs) { return formatAttributeValue(attrs); },
             attrs);
}
static std::size_t hashAttrs(const OpAttrs& attrs) {
  std::size_t hash = attrs.index();
  std::visit([&](const auto& value) { hashAttributeValue(hash, value); },
             attrs);
  return hash;
}
bool TensorNode::matches(const TensorNode& other) const {
  return op == other.op && attrs == other.attrs &&
         operands.size() == other.operands.size();
}
bool TensorNode::operator==(const TensorNode& other) const {
  return op == other.op && attrs == other.attrs && operands == other.operands;
}
std::size_t TensorNode::hash() const {
  std::size_t result = std::hash<OpKind>{}(op);
  eggc::hash_combine(result, hashAttrs(attrs));
  for (eggc::Id child : operands)
    eggc::hash_combine(result, std::hash<eggc::Id>{}(child));
  return result;
}

namespace {
constexpr OpSchema schemas[] = {
#define TENSOR_VALUE(record, fields)
#define TENSOR_ATTRS(record, fields)
#define TENSOR_OP(kind, callable, external, operands, record)         \
  {OpKind::kind, external, operands, std::is_same_v<record, NoAttrs>, \
   [](const OpAttrs& attrs) {                                         \
     return std::holds_alternative<record>(attrs);                    \
   }},
#include "research/joint_shard/bridge/tensor_lang/tensorlang.def"
#undef TENSOR_OP
#undef TENSOR_ATTRS
#undef TENSOR_VALUE
};
}  // namespace
std::span<const OpSchema> opSchemas() { return schemas; }
const OpSchema* opSchema(OpKind op) {
  for (const auto& schema : schemas)
    if (schema.op == op) return &schema;
  return nullptr;
}
const char* stableHloName(OpKind op) {
  if (const auto* schema = opSchema(op)) return schema->name.data();
  throw std::invalid_argument("unknown TensorLang operator kind");
}
const OpSchema* lookupOpSchema(std::string_view name) {
  for (const auto& schema : schemas)
    if (schema.name == name) return &schema;
  return nullptr;
}
bool validNodeSchema(const TensorNode& node) {
  const auto* schema = opSchema(node.op);
  if (!schema || schema->arity != node.operands.size()) return false;
  return schema->accepts_attributes(node.attrs);
}

}  // namespace joint_shard

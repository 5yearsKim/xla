#include "research/joint_shard/bridge/tensor_lang/tensorlang.h"

#include <type_traits>

#include "llvm/Support/raw_ostream.h"

namespace {
template <class Handle>
void hashHandle(std::size_t& seed, Handle handle) {
  eggc::hash_combine(seed, std::hash<const void*>{}(
                               handle ? handle.getAsOpaquePointer() : nullptr));
}
void hashDimensions(std::size_t& seed, const std::vector<int64_t>& dims) {
  eggc::hash_combine(seed, dims.size());
  for (int64_t dim : dims) eggc::hash_combine(seed, std::hash<int64_t>{}(dim));
}
void printDimensions(llvm::raw_ostream& out, const std::vector<int64_t>& dims) {
  out << '[';
  for (std::size_t i = 0; i < dims.size(); ++i) {
    if (i) out << ',';
    out << dims[i];
  }
  out << ']';
}
}  // namespace
static std::size_t hashAttrs(const OpAttrs& attrs) {
  std::size_t hash = attrs.index();
  std::visit(
      [&](const auto& value) {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, InputAttrs>) {
          eggc::hash_combine(hash, value.index);
          hashHandle(hash, value.type);
        } else if constexpr (std::is_same_v<T, DotGeneralAttrs>) {
          hashDimensions(hash, value.lhs_contracting);
          hashDimensions(hash, value.rhs_contracting);
          hashDimensions(hash, value.lhs_batching);
          hashDimensions(hash, value.rhs_batching);
          hashHandle(hash, value.precision_config);
          hashHandle(hash, value.algorithm);
          hashHandle(hash, value.result_type);
          hashHandle(hash, value.extra_attributes);
        } else if constexpr (std::is_same_v<T, TransposeAttrs>) {
          hashDimensions(hash, value.permutation);
        } else if constexpr (std::is_same_v<T, ReshapeAttrs>) {
          hashHandle(hash, value.result_type);
        } else if constexpr (std::is_same_v<T, BroadcastAttrs>) {
          hashDimensions(hash, value.dimensions);
          hashHandle(hash, value.result_type);
        } else if constexpr (std::is_same_v<T, ConstantAttrs>) {
          hashHandle(hash, value.value);
        } else if constexpr (std::is_same_v<T, ReduceAttrs>) {
          eggc::hash_combine(hash, std::hash<ReduceKind>{}(value.kind));
          hashDimensions(hash, value.axes);
          hashHandle(hash, value.initializer);
        }
      },
      attrs);
  return hash;
}
std::string TensorNode::format() const {
  std::string result;
  llvm::raw_string_ostream out(result);
  out << stableHloName(op);
  std::visit(
      [&](const auto& value) {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, InputAttrs>) {
          out << "{index=" << value.index << ",type=" << value.type << '}';
        } else if constexpr (std::is_same_v<T, DotGeneralAttrs>) {
          out << "{lhs_contracting=";
          printDimensions(out, value.lhs_contracting);
          out << ",rhs_contracting=";
          printDimensions(out, value.rhs_contracting);
          out << ",lhs_batching=";
          printDimensions(out, value.lhs_batching);
          out << ",rhs_batching=";
          printDimensions(out, value.rhs_batching);
          if (value.precision_config)
            out << ",precision=" << value.precision_config;
          if (value.algorithm) out << ",algorithm=" << value.algorithm;
          if (value.result_type) out << ",result_type=" << value.result_type;
          if (value.extra_attributes)
            out << ",extra=" << value.extra_attributes;
          out << '}';
        } else if constexpr (std::is_same_v<T, TransposeAttrs>) {
          out << "{permutation=";
          printDimensions(out, value.permutation);
          out << '}';
        } else if constexpr (std::is_same_v<T, ReshapeAttrs>) {
          out << "{result_type=" << value.result_type << '}';
        } else if constexpr (std::is_same_v<T, BroadcastAttrs>) {
          out << "{dimensions=";
          printDimensions(out, value.dimensions);
          out << ",result_type=" << value.result_type << '}';
        } else if constexpr (std::is_same_v<T, ConstantAttrs>) {
          out << "{value=" << value.value << '}';
        } else if constexpr (std::is_same_v<T, ReduceAttrs>) {
          out << "{kind=" << static_cast<int>(value.kind) << ",axes=";
          printDimensions(out, value.axes);
          if (value.initializer) out << ",init=" << value.initializer;
          out << '}';
        }
      },
      attrs);
  return result;
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
    {OpKind::Input, "input", 0, false},
    {OpKind::Constant, "constant", 0, false},
    {OpKind::Add, "add", 2, true},
    {OpKind::Subtract, "subtract", 2, true},
    {OpKind::Multiply, "multiply", 2, true},
    {OpKind::Divide, "divide", 2, true},
    {OpKind::Maximum, "maximum", 2, true},
    {OpKind::Minimum, "minimum", 2, true},
    {OpKind::Negate, "negate", 1, true},
    {OpKind::Exp, "exp", 1, true},
    {OpKind::Log, "log", 1, true},
    {OpKind::Sqrt, "sqrt", 1, true},
    {OpKind::Tanh, "tanh", 1, true},
    {OpKind::DotGeneral, "dot_general", 2, false},
    {OpKind::Reduce, "reduce", 1, false},
    {OpKind::Transpose, "transpose", 1, false},
    {OpKind::Reshape, "reshape", 1, false},
    {OpKind::BroadcastInDim, "broadcast_in_dim", 1, false},
};
}
const OpSchema* opSchema(OpKind op) {
  for (const auto& schema : schemas)
    if (schema.op == op) return &schema;
  return nullptr;
}
const OpSchema* lookupOpSchema(std::string_view name) {
  for (const auto& schema : schemas)
    if (schema.name == name || stableHloName(schema.op) == name) return &schema;
  return nullptr;
}
bool validNodeSchema(const TensorNode& node) {
  const auto* schema = opSchema(node.op);
  if (!schema || schema->arity != node.operands.size()) return false;
  if (schema->attribute_free)
    return std::holds_alternative<NoAttrs>(node.attrs);
  switch (node.op) {
    case OpKind::Input:
      return std::holds_alternative<InputAttrs>(node.attrs);
    case OpKind::Constant:
      return std::holds_alternative<ConstantAttrs>(node.attrs);
    case OpKind::DotGeneral:
      return std::holds_alternative<DotGeneralAttrs>(node.attrs);
    case OpKind::Transpose:
      return std::holds_alternative<TransposeAttrs>(node.attrs);
    case OpKind::Reshape:
      return std::holds_alternative<ReshapeAttrs>(node.attrs);
    case OpKind::BroadcastInDim:
      return std::holds_alternative<BroadcastAttrs>(node.attrs);
    case OpKind::Reduce:
      return std::holds_alternative<ReduceAttrs>(node.attrs);
    default:
      return false;
  }
}

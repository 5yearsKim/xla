#include "research/joint_shard/bridge/tensor_lang/op_properties.h"

#include <algorithm>

#include "llvm/Support/Casting.h"

NumericalPermissions numericalPermissions(NumericalPolicy policy) {
  if (policy == NumericalPolicy::AllowReassociation)
    return {true, true, true, true, true, true};
  return {};
}

std::span<const OpProperty> declaredProperties(OpKind op) {
  switch (op) {
    case OpKind::Add:
    case OpKind::Maximum:
    case OpKind::Minimum: {
      static const OpProperty properties[] = {Elementwise{}, Commutative{},
                                              Associative{}};
      return properties;
    }
    case OpKind::Multiply: {
      static const OpProperty properties[] = {
          Elementwise{}, Commutative{},    Associative{},   LinearIn{0},
          LinearIn{1},   HomogeneousIn{0}, HomogeneousIn{1}};
      return properties;
    }
    case OpKind::DotGeneral: {
      static const OpProperty properties[] = {
          LinearIn{0}, LinearIn{1}, HomogeneousIn{0}, HomogeneousIn{1}};
      return properties;
    }
    case OpKind::Reduce: {
      static const OpProperty properties[] = {LinearIn{0}, HomogeneousIn{0}};
      return properties;
    }
    case OpKind::Negate: {
      static const OpProperty properties[] = {Elementwise{}, Involution{},
                                              LinearIn{0}, HomogeneousIn{0}};
      return properties;
    }
    case OpKind::Transpose: {
      static const OpProperty properties[] = {Involution{}, LinearIn{0},
                                              HomogeneousIn{0}};
      return properties;
    }
    case OpKind::Reshape:
    case OpKind::BroadcastInDim: {
      static const OpProperty properties[] = {LinearIn{0}, HomogeneousIn{0}};
      return properties;
    }
    case OpKind::Subtract:
    case OpKind::Divide:
    case OpKind::Exp:
    case OpKind::Log:
    case OpKind::Sqrt:
    case OpKind::Tanh: {
      static const OpProperty properties[] = {Elementwise{}};
      return properties;
    }
    default:
      return {};
  }
}

PropertyDecision queryProperty(const TensorNode& node,
                               const OpProperty& property,
                               const PropertyContext& context) {
  const auto properties = declaredProperties(node.op);
  if (std::find(properties.begin(), properties.end(), property) ==
      properties.end())
    return {false, "operator does not declare the property"};
  const auto inference = inferTensorNode(node, context.operands);
  if (!inference.valid() || !context.result.type)
    return {false, inference.reason.empty() ? "unknown result facts"
                                            : inference.reason};
  if (inference.facts.type != context.result.type)
    return {false, "result type mismatch"};
  if (std::holds_alternative<Elementwise>(property)) return {true, {}};
  if (std::holds_alternative<Involution>(property) &&
      node.op == OpKind::Transpose) {
    const auto& p = std::get<TransposeAttrs>(node.attrs).permutation;
    for (size_t i = 0; i < p.size(); ++i)
      if (p[p[i]] != static_cast<int64_t>(i))
        return {false, "permutation is not self-inverse"};
    return {true, {}};
  }
  if (node.op == OpKind::Reduce &&
      std::get<ReduceAttrs>(node.attrs).kind != ReduceKind::Sum)
    return {false,
            "only canonical sum reduction declares this linear contract"};
  // Moving data preserves each element without changing arithmetic.
  if (node.op == OpKind::Transpose || node.op == OpKind::Reshape ||
      node.op == OpKind::BroadcastInDim) {
    if (!context.result.type.hasStaticShape())
      return {false, "layout algebra requires static shapes"};
    for (const auto& operand : context.operands)
      if (!operand.type.hasStaticShape())
        return {false, "layout algebra requires static shapes"};
    return {true, {}};
  }
  if (!context.result.type.hasStaticShape())
    return {false, "requires known static shape"};
  for (const auto& operand : context.operands)
    if (!operand.type.hasStaticShape())
      return {false, "requires static operand shapes"};
  const auto element = context.result.type.getElementType();
  const bool exact = llvm::isa<mlir::IntegerType>(element);
  const auto permissions =
      context.permissions.value_or(numericalPermissions(context.policy));
  if (node.op == OpKind::DotGeneral) {
    const auto& dot = std::get<DotGeneralAttrs>(node.attrs);
    if (dot.algorithm ||
        (dot.extra_attributes && !dot.extra_attributes.empty()))
      return {false,
              "dot algorithm or unknown attributes lack an algebraic contract"};
    // Integer dots are exact modular arithmetic. Floating dot accumulation and
    // precision lowering require the caller's explicit relaxed permission.
    if (!exact && !permissions.rewrite_dot_arithmetic)
      return {false, "floating dot arithmetic requires relaxed policy"};
  }
  if (exact) return {true, {}};
  if (!permissions.assume_finite || !permissions.ignore_signed_zero)
    return {
        false,
        "floating algebra requires finite-value and signed-zero permissions"};
  if (std::holds_alternative<Commutative>(property) &&
      !permissions.reorder_floating_point)
    return {false, "floating operand reordering is disabled"};
  if (std::holds_alternative<Associative>(property) &&
      !permissions.reassociate_floating_point)
    return {false, "floating reassociation is disabled"};
  if ((std::holds_alternative<LinearIn>(property) ||
       std::holds_alternative<HomogeneousIn>(property)) &&
      !permissions.distribute_floating_point)
    return {false, "floating distribution is disabled"};
  return {true, {}};
}
bool hasProperty(const TensorNode& node, const OpProperty& property,
                 const PropertyContext& context) {
  return queryProperty(node, property, context).allowed;
}

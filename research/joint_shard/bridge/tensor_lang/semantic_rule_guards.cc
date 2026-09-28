#include <stdexcept>

#include "llvm/Support/Casting.h"
#include "research/joint_shard/bridge/tensor_lang/semantic_rule_internal.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace joint_shard::semantic_detail {
namespace {
PropertyDecision checkArithmetic(const TensorNode& node, TensorFacts result,
                                 std::span<const TensorFacts> operands,
                                 NumericalPermissions permissions) {
  if (!result.type || !result.type.hasStaticShape())
    return {false, "dot rule requires a static result shape"};
  for (const auto& operand : operands)
    if (!operand.type || !operand.type.hasStaticShape())
      return {false, "dot rule requires static operand shapes"};
  if (node.op == OpKind::DotGeneral) {
    const auto& dot = std::get<DotGeneralAttrs>(node.attrs);
    if (dot.algorithm ||
        (dot.extra_attributes && !dot.extra_attributes.empty()))
      return {false, "dot algorithm or extra attributes are unsupported"};
    if (dot.precision_config) {
      if (dot.precision_config.size() != 0 && dot.precision_config.size() != 2)
        return {false, "unsupported dot precision configuration"};
      for (auto attr : dot.precision_config) {
        auto precision = llvm::dyn_cast<mlir::stablehlo::PrecisionAttr>(attr);
        if (!precision ||
            precision.getValue() != mlir::stablehlo::Precision::DEFAULT)
          return {false, "dot rule supports only default precision"};
      }
    }
  }
  auto element = result.type.getElementType();
  if (auto integer = llvm::dyn_cast<mlir::IntegerType>(element)) {
    if (integer.getWidth() > 1) return {true, {}};
  }
  if (!llvm::isa<mlir::FloatType>(element))
    return {false,
            "dot rule supports real floating or modular integer tensors"};
  if (!permissions.rewrite_dot_arithmetic)
    return {false, "floating dot arithmetic is disabled"};
  if (!permissions.reassociate_floating_point ||
      !permissions.distribute_floating_point)
    return {false, "dot rule requires floating reassociation and distribution"};
  if (!permissions.assume_finite || !permissions.ignore_signed_zero)
    return {false,
            "dot rule requires finite-value and signed-zero permissions"};
  return {true, {}};
}
}  // namespace

PropertyDecision checkRuleGuard(const Predicate& predicate,
                                const Bindings& bindings, const Prepared& rhs,
                                const TensorEGraph& graph,
                                NumericalPolicy policy,
                                const SemanticRuleOptions& options) {
  if (!predicate.isRuleGuard()) throw std::logic_error("not a rule guard");
  auto permissions = options.permissions.value_or(numericalPermissions(policy));
  if (predicate.kind == PredicateKind::RawMoments &&
      (!permissions.rewrite_raw_moments ||
       !permissions.reassociate_floating_point ||
       !permissions.distribute_floating_point || !permissions.assume_finite ||
       !permissions.ignore_signed_zero))
    return {false, "raw moments require aggressive floating permissions"};
  size_t lhs_dots = 0, lhs_sums = 0, rhs_dots = 0, rhs_sums = 0;
  const auto check = [&](const TensorNode& node, TensorFacts result,
                         std::span<const TensorFacts> operands, size_t& dots,
                         size_t& sums) -> PropertyDecision {
    if (predicate.kind == PredicateKind::DotDivision &&
        node.op == OpKind::Divide) {
      if (!result.type ||
          !llvm::isa<mlir::FloatType>(result.type.getElementType()))
        return {false, "dot division requires real floating tensors"};
      if (!permissions.rewrite_dot_division)
        return {false, "floating dot division is disabled"};
    }
    if (node.op != OpKind::DotGeneral && node.op != OpKind::Reduce)
      return {true, {}};
    if (predicate.kind == PredicateKind::RawMoments &&
        (!result.type ||
         !llvm::isa<mlir::FloatType>(result.type.getElementType())))
      return {false, "raw moments require real floating tensors"};
    if (node.op == OpKind::DotGeneral)
      ++dots;
    else {
      if (std::get<ReduceAttrs>(node.attrs).kind != ReduceKind::Sum)
        return {false, "sum/dot rule requires canonical sum reductions"};
      ++sums;
    }
    if (predicate.kind == PredicateKind::RawMoments) {
      if (!result.type.hasStaticShape())
        return {false, "raw moments require static shapes"};
      for (const auto& operand : operands)
        if (!operand.type || !operand.type.hasStaticShape())
          return {false, "raw moments require static shapes"};
      return {true, {}};
    }
    return checkArithmetic(node, result, operands, permissions);
  };
  const auto checkOccurrence = [&](const OperatorOccurrence& occurrence) {
    std::vector<TensorFacts> operands;
    for (auto child : occurrence.node.operands)
      operands.push_back(tensorFacts(graph, child));
    return check(occurrence.node, tensorFacts(graph, occurrence.eclass),
                 operands, lhs_dots, lhs_sums);
  };
  for (const auto& occurrence : bindings.concrete) {
    auto decision = checkOccurrence(occurrence);
    if (!decision.allowed) return decision;
  }
  for (const auto& [name, occurrences] : bindings.operators)
    for (const auto& occurrence : occurrences) {
      auto decision = checkOccurrence(occurrence);
      if (!decision.allowed) return decision;
    }
  const auto visit = [&](auto&& self, const Prepared& p) -> PropertyDecision {
    if (p.reference) return {true, {}};
    std::vector<TensorFacts> operands;
    for (const auto& child : p.children) {
      auto decision = self(self, child);
      if (!decision.allowed) return decision;
      operands.push_back(child.facts);
    }
    return check(p.node, p.facts, operands, rhs_dots, rhs_sums);
  };
  auto decision = visit(visit, rhs);
  if (!decision.allowed) return decision;
  if (predicate.kind == PredicateKind::RawMoments) {
    if (!lhs_sums || !rhs_sums)
      return {false, "raw moments require sum reductions on both sides"};
  } else if (predicate.kind == PredicateKind::DotReassociation) {
    if (lhs_dots < 2 || rhs_dots < 2)
      return {false, "dot reassociation requires two dots on each side"};
  } else if (predicate.kind == PredicateKind::DotArithmetic ||
             predicate.kind == PredicateKind::DotDivision) {
    if (!lhs_dots || !rhs_dots)
      return {false, "dot arithmetic requires a dot on each side"};
  } else if (!lhs_dots || !rhs_dots || !lhs_sums || !rhs_sums) {
    return {false, "sum/dot interchange requires a sum and dot on each side"};
  }
  return {true, {}};
}
}  // namespace joint_shard::semantic_detail

#include "research/joint_shard/bridge/stablehlo_exporter.h"

#include <stdexcept>

#include "mlir/IR/OperationSupport.h"
#include "research/joint_shard/bridge/tensor_lang/tensor_analysis.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace joint_shard {

mlir::Value StableHloExporter::exportExpr(const TensorRecExpr& expr,
                                          std::size_t node) {
  return exportRoots(expr, {node}).front();
}
std::vector<mlir::Value> StableHloExporter::exportRoots(
    const TensorRecExpr& expr, const std::vector<std::size_t>& roots) {
  // Validate the complete expression before emitting any MLIR operations.
  std::vector<TensorFacts> facts;
  for (size_t index = 0; index < expr.nodes.size(); ++index) {
    const auto& node = expr.nodes[index];
    std::vector<TensorFacts> operands;
    for (auto child : node.operands) {
      if (child >= index)
        throw std::invalid_argument("expression children must precede parent");
      operands.push_back(facts[child]);
    }
    auto inference = inferTensorNode(node, operands);
    if (!inference.valid())
      throw std::invalid_argument("cannot export TensorLang: " +
                                  inference.reason);
    if (node.op == OpKind::Input) {
      const auto& input = std::get<InputAttrs>(node.attrs);
      if (input.index >= inputs_.size() ||
          inputs_[input.index].getType() != input.type)
        throw std::invalid_argument("region input index or type mismatch");
    }
    facts.push_back(inference.facts);
  }
  for (auto root : roots)
    if (root >= facts.size())
      throw std::out_of_range("expression root out of range");
  values_.assign(expr.nodes.size(), {});
  types_.clear();
  for (const auto& fact : facts) types_.push_back(fact.type);
  std::vector<mlir::Value> results;
  for (auto root : roots) results.push_back(exportNode(expr, root));
  return results;
}
mlir::Value StableHloExporter::exportNode(const TensorRecExpr& expr,
                                          std::size_t index) {
  if (values_[index]) return values_[index];
  const auto& node = expr.nodes[index];
  if (node.op == OpKind::Input)
    return values_[index] = inputs_[std::get<InputAttrs>(node.attrs).index];
  std::vector<mlir::Value> operands;
  for (auto child : node.operands) operands.push_back(exportNode(expr, child));
  mlir::OperationState state(loc_, stableHloName(node.op));
  state.addTypes(types_[index]);
  if (auto* attrs = std::get_if<ConstantAttrs>(&node.attrs))
    state.addAttribute("value", attrs->value);
  if (auto* attrs = std::get_if<TransposeAttrs>(&node.attrs))
    state.addAttribute("permutation",
                       builder_.getDenseI64ArrayAttr(attrs->permutation));
  if (auto* attrs = std::get_if<BroadcastAttrs>(&node.attrs))
    state.addAttribute("broadcast_dimensions",
                       builder_.getDenseI64ArrayAttr(attrs->dimensions));
  if (auto* attrs = std::get_if<DotGeneralAttrs>(&node.attrs)) {
    if (attrs->extra_attributes)
      state.addAttributes(attrs->extra_attributes.getValue());
    state.addAttribute(
        "dot_dimension_numbers",
        mlir::stablehlo::DotDimensionNumbersAttr::get(
            builder_.getContext(), attrs->lhs_batching, attrs->rhs_batching,
            attrs->lhs_contracting, attrs->rhs_contracting));
    if (attrs->precision_config)
      state.addAttribute("precision_config", attrs->precision_config);
    if (attrs->algorithm) state.addAttribute("algorithm", attrs->algorithm);
  }
  if (auto* attrs = std::get_if<ReduceAttrs>(&node.attrs)) {
    mlir::OperationState init(loc_, "stablehlo.constant");
    init.addTypes(attrs->initializer.getType());
    init.addAttribute("value", attrs->initializer);
    operands.push_back(builder_.create(init)->getResult(0));
    state.addAttribute("dimensions",
                       builder_.getDenseI64ArrayAttr(attrs->axes));
    auto* body = state.addRegion();
    auto* block = new mlir::Block;
    body->push_back(block);
    auto scalar = attrs->initializer.getType();
    block->addArgument(scalar, loc_);
    block->addArgument(scalar, loc_);
    mlir::OpBuilder::InsertionGuard guard(builder_);
    builder_.setInsertionPointToStart(block);
    const char* name = "stablehlo.add";
    switch (attrs->kind) {
      case ReduceKind::Sum:
        break;
      case ReduceKind::Product:
        name = "stablehlo.multiply";
        break;
      case ReduceKind::Max:
        name = "stablehlo.maximum";
        break;
      case ReduceKind::Min:
        name = "stablehlo.minimum";
        break;
    }
    mlir::OperationState reducer(loc_, name);
    reducer.addOperands(block->getArguments());
    reducer.addTypes(scalar);
    auto value = builder_.create(reducer)->getResult(0);
    mlir::OperationState ret(loc_, "stablehlo.return");
    ret.addOperands(value);
    builder_.create(ret);
  }
  state.addOperands(operands);
  return values_[index] = builder_.create(state)->getResult(0);
}

}  // namespace joint_shard

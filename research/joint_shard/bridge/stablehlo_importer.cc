#include "research/joint_shard/bridge/stablehlo_importer.h"

#include <algorithm>
#include <optional>
#include <stdexcept>

#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace joint_shard {

namespace {
// Derived XLA layout/debug hints are deliberately recomputed by downstream
// lowering. Unknown metadata and shardings remain hard rewrite boundaries.
bool knownAttributes(mlir::Operation* op,
                     std::initializer_list<llvm::StringRef> semantic) {
  for (auto attr : op->getAttrs()) {
    auto name = attr.getName().getValue();
    if (name == "result_layout" || name == "xla_shape") continue;
    if (std::find(semantic.begin(), semantic.end(), name) == semantic.end())
      return false;
  }
  return true;
}
std::vector<int64_t> dimensions(mlir::Operation* op, llvm::StringRef name) {
  auto attr = op->getAttrOfType<mlir::DenseI64ArrayAttr>(name);
  if (!attr)
    throw std::invalid_argument("missing dimension attribute: " + name.str());
  return {attr.asArrayRef().begin(), attr.asArrayRef().end()};
}
std::optional<ReduceAttrs> reduction(mlir::Operation* op) {
  if (op->getNumOperands() != 2 || op->getNumRegions() != 1 ||
      !knownAttributes(op, {"dimensions"}) ||
      !llvm::hasSingleElement(op->getRegion(0)))
    return {};
  auto& block = op->getRegion(0).front();
  if (block.getNumArguments() != 2 || block.getOperations().size() != 2)
    return {};
  auto* reducer = &block.front();
  auto* ret = &block.back();
  if (reducer->getNumOperands() != 2 || reducer->getNumResults() != 1 ||
      reducer->getNumRegions() || !reducer->getAttrs().empty() ||
      reducer->getOperand(0) != block.getArgument(0) ||
      reducer->getOperand(1) != block.getArgument(1) ||
      ret->getName().getStringRef() != "stablehlo.return" ||
      ret->getNumOperands() != 1 || ret->getNumResults() != 0 ||
      ret->getNumRegions() != 0 || !ret->getAttrs().empty() ||
      ret->getOperand(0) != reducer->getResult(0))
    return {};
  ReduceKind kind;
  const auto name = reducer->getName().getStringRef();
  if (name == "stablehlo.add")
    kind = ReduceKind::Sum;
  else if (name == "stablehlo.multiply")
    kind = ReduceKind::Product;
  else if (name == "stablehlo.maximum")
    kind = ReduceKind::Max;
  else if (name == "stablehlo.minimum")
    kind = ReduceKind::Min;
  else
    return {};
  auto* init = op->getOperand(1).getDefiningOp();
  if (!init ||
      (init->getName().getStringRef() != "stablehlo.constant" &&
       init->getName().getStringRef() != "sdy.constant") ||
      !knownAttributes(init, {"value"}))
    return {};
  auto value = init->getAttrOfType<mlir::ElementsAttr>("value");
  auto input =
      llvm::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
  if (!input || !value) return {};
  auto scalar = mlir::RankedTensorType::get({}, input.getElementType());
  if (block.getArgument(0).getType() != scalar ||
      block.getArgument(1).getType() != scalar ||
      reducer->getResult(0).getType() != scalar ||
      op->getOperand(1).getType() != scalar)
    return {};
  ReduceAttrs attrs{kind, dimensions(op, "dimensions"), value};
  if (!canonicalReductionIdentity(attrs, input.getElementType())) return {};
  return attrs;
}
std::optional<TensorNode> importNode(mlir::Operation* op) {
  if (!op || op->getNumResults() != 1 ||
      !llvm::isa<mlir::RankedTensorType>(op->getResult(0).getType()))
    return {};
  for (auto value : op->getOperands())
    if (!llvm::isa<mlir::RankedTensorType>(value.getType())) return {};
  const auto name = op->getName().getStringRef();
  auto result = llvm::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (name == "stablehlo.reduce") {
    auto attrs = reduction(op);
    if (!attrs) return {};
    return TensorNode{OpKind::Reduce, *attrs, {0}};
  }
  if (op->getNumRegions()) return {};
  if (name == "sdy.constant" || name == "stablehlo.constant") {
    if (!knownAttributes(op, {"value"})) return {};
    auto value = op->getAttrOfType<mlir::ElementsAttr>("value");
    if (!value) return {};
    return TensorNode{OpKind::Constant, ConstantAttrs{value}, {}};
  }
  if (name == "stablehlo.transpose") {
    if (!knownAttributes(op, {"permutation"})) return {};
    return TensorNode{
        OpKind::Transpose, TransposeAttrs{dimensions(op, "permutation")}, {0}};
  }
  if (name == "stablehlo.reshape") {
    if (!knownAttributes(op, {})) return {};
    return TensorNode{OpKind::Reshape, ReshapeAttrs{result}, {0}};
  }
  if (name == "stablehlo.broadcast_in_dim") {
    if (!knownAttributes(op, {"broadcast_dimensions"})) return {};
    return TensorNode{
        OpKind::BroadcastInDim,
        BroadcastAttrs{dimensions(op, "broadcast_dimensions"), result},
        {0}};
  }
  if (auto dot = llvm::dyn_cast<mlir::stablehlo::DotGeneralOp>(op)) {
    if (!knownAttributes(
            op, {"dot_dimension_numbers", "precision_config", "algorithm"}))
      return {};
    const auto dims = dot.getDotDimensionNumbers();
    return TensorNode{
        OpKind::DotGeneral,
        DotGeneralAttrs{{dims.getLhsContractingDimensions().begin(),
                         dims.getLhsContractingDimensions().end()},
                        {dims.getRhsContractingDimensions().begin(),
                         dims.getRhsContractingDimensions().end()},
                        {dims.getLhsBatchingDimensions().begin(),
                         dims.getLhsBatchingDimensions().end()},
                        {dims.getRhsBatchingDimensions().begin(),
                         dims.getRhsBatchingDimensions().end()},
                        op->getAttrOfType<mlir::ArrayAttr>("precision_config"),
                        op->getAttr("algorithm"),
                        result,
                        mlir::DictionaryAttr::get(op->getContext())},
        {0, 0}};
  }
  const OpSchema* schema = lookupOpSchema(name.str());
  // StableHLO calls this exponential; TensorLang's DSL uses exp.
  if (name == "stablehlo.exponential") schema = opSchema(OpKind::Exp);
  if (!schema || !schema->attribute_free || !knownAttributes(op, {})) return {};
  return TensorNode{schema->op, NoAttrs{},
                    std::vector<eggc::Id>(schema->arity, 0)};
}
}  // namespace

std::string tensorImportRejection(mlir::Operation* op) {
  try {
    auto node = importNode(op);
    if (!node) return "unsupported operator, region, or metadata";
    const unsigned count =
        node->op == OpKind::Reduce ? 1 : op->getNumOperands();
    if (count != node->operands.size()) return "operand arity mismatch";
    std::vector<TensorFacts> facts;
    for (unsigned i = 0; i < count; ++i)
      facts.push_back(
          {llvm::cast<mlir::RankedTensorType>(op->getOperand(i).getType()),
           {}});
    auto inferred = inferTensorNode(*node, facts);
    if (!inferred.valid()) return inferred.reason;
    if (inferred.facts.type != op->getResult(0).getType())
      return "inferred result type differs from MLIR";
    return {};
  } catch (const std::exception& error) {
    return error.what();
  }
}
bool canImportTensorOperation(mlir::Operation* op) {
  return tensorImportRejection(op).empty();
}
void StableHloImporter::bindValue(mlir::Value value, unsigned index) {
  auto type = llvm::dyn_cast<mlir::RankedTensorType>(value.getType());
  if (!type)
    throw std::invalid_argument("region input requires ranked tensor type");
  cache_[value] =
      graph_.add(TensorNode{OpKind::Input, InputAttrs{index, type}, {}});
  original_ids_[value] =
      original_.add(TensorNode{OpKind::Input, InputAttrs{index, type}, {}});
}
eggc::Id StableHloImporter::importValue(mlir::Value value) {
  auto cached = cache_.find(value);
  if (cached != cache_.end()) return cached->second;
  auto* op = value.getDefiningOp();
  auto rejection = tensorImportRejection(op);
  if (!rejection.empty())
    throw std::invalid_argument("bind unsupported value as input: " +
                                rejection);
  auto node = *importNode(op);
  node.operands.clear();
  const unsigned count = node.op == OpKind::Reduce ? 1 : op->getNumOperands();
  for (unsigned i = 0; i < count; ++i)
    node.operands.push_back(importValue(op->getOperand(i)));
  auto original = node;
  for (unsigned i = 0; i < count; ++i)
    original.operands[i] = original_ids_.lookup(op->getOperand(i));
  original_ids_[value] = original_.add(std::move(original));
  auto id = graph_.add(std::move(node));
  cache_[value] = id;
  return id;
}

}  // namespace joint_shard

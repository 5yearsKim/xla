#include "research/joint_shard/stablehlo_importer.h"

#include <stdexcept>
#include <string>
#include <utility>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "stablehlo/dialect/StablehloOps.h"

eggc::Id StableHloImporter::importValue(mlir::Value value) {
  auto cached = cache_.find(value);
  if (cached != cache_.end()) return cached->second;

  if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(value)) {
    eggc::Id id = graph_.add("arg" + std::to_string(argument.getArgNumber()));
    cache_[value] = id;
    return id;
  }

  mlir::Operation* op = value.getDefiningOp();
  if (auto dot = llvm::dyn_cast<mlir::stablehlo::DotGeneralOp>(op)) {
    auto lhsType =
        llvm::dyn_cast<mlir::RankedTensorType>(dot.getLhs().getType());
    auto rhsType =
        llvm::dyn_cast<mlir::RankedTensorType>(dot.getRhs().getType());
    auto dims = dot.getDotDimensionNumbers();
    if (!descriptors_ || !lhsType || !rhsType || lhsType.getRank() != 2 ||
        rhsType.getRank() != 2 || !dims.getLhsBatchingDimensions().empty() ||
        !dims.getRhsBatchingDimensions().empty() ||
        dims.getLhsContractingDimensions().size() != 1 ||
        dims.getRhsContractingDimensions().size() != 1 ||
        dims.getLhsContractingDimensions()[0] != 1 ||
        dims.getRhsContractingDimensions()[0] != 0) {
      throw std::runtime_error(
          "dot_general requires a descriptor table and rank-2 matmul with "
          "contracting dimensions [1] x [0], without batching");
    }
    std::string name =
        descriptors_->internDot({lhsType, rhsType, dot.getResult().getType(),
                                 dot->getAttrDictionary()});
    eggc::Id id = graph_.add(
        name, {importValue(dot.getLhs()), importValue(dot.getRhs())});
    cache_[value] = id;
    return id;
  }
  if (auto add = llvm::dyn_cast<mlir::stablehlo::AddOp>(op)) {
    eggc::Id id = graph_.add(
        "add", {importValue(add.getLhs()), importValue(add.getRhs())});
    cache_[value] = id;
    return id;
  }

  if (auto multiply = llvm::dyn_cast<mlir::stablehlo::MulOp>(op)) {
    eggc::Id id = graph_.add("multiply", {importValue(multiply.getLhs()),
                                          importValue(multiply.getRhs())});
    cache_[value] = id;
    return id;
  }

  if (auto exponential = llvm::dyn_cast<mlir::stablehlo::ExpOp>(op)) {
    eggc::Id id = graph_.add("exp", {importValue(exponential.getOperand())});
    cache_[value] = id;
    return id;
  }

  throw std::runtime_error("Unsupported StableHLO value in importer");
}

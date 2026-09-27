#ifndef RESEARCH_JOINT_SHARD_BRIDGE_STABLEHLO_IMPORTER_H_
#define RESEARCH_JOINT_SHARD_BRIDGE_STABLEHLO_IMPORTER_H_

#include <string>

#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/Value.h"
#include "eggc/egraph.hpp"
#include "research/joint_shard/bridge/tensor_lang/tensor_analysis.h"

// Shared admission policy used by the region splitter and importer.
// Unsupported and annotated operations stay in MLIR as opaque boundaries.
// Only canonical single-input reducer regions can be imported.
std::string tensorImportRejection(mlir::Operation* op);
bool canImportTensorOperation(mlir::Operation* op);

class StableHloImporter {
 public:
  explicit StableHloImporter(TensorEGraph& graph) : graph_(graph) {}

  // Bind every region boundary before importValue, including block arguments.
  void bindValue(mlir::Value value, unsigned index);
  eggc::Id importValue(mlir::Value value);
  // Recorded directly from the source SSA graph, before equality saturation.
  const TensorRecExpr& originalExpression() const { return original_; }
  size_t originalRoot(mlir::Value value) const {
    return original_ids_.lookup(value);
  }

 private:
  TensorEGraph& graph_;
  llvm::DenseMap<mlir::Value, eggc::Id> cache_;
  TensorRecExpr original_;
  llvm::DenseMap<mlir::Value, eggc::Id> original_ids_;
};

#endif  // RESEARCH_JOINT_SHARD_BRIDGE_STABLEHLO_IMPORTER_H_

#ifndef RESEARCH_JOINT_SHARD_BRIDGE_STABLEHLO_IMPORTER_H_
#define RESEARCH_JOINT_SHARD_BRIDGE_STABLEHLO_IMPORTER_H_

#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/Value.h"
#include "eggc/egraph.hpp"
#include "research/joint_shard/bridge/operation_descriptors.h"

class StableHloImporter {
 public:
  explicit StableHloImporter(eggc::EGraph& graph,
                             OperationDescriptors* descriptors = nullptr)
      : graph_(graph), descriptors_(descriptors) {}

  // Bind a region boundary to an opaque leaf shared with the exporter.
  void bindValue(mlir::Value value, unsigned index);
  eggc::Id importValue(mlir::Value value);

 private:
  eggc::EGraph& graph_;
  OperationDescriptors* descriptors_;
  llvm::DenseMap<mlir::Value, eggc::Id> cache_;
};

#endif  // RESEARCH_JOINT_SHARD_BRIDGE_STABLEHLO_IMPORTER_H_

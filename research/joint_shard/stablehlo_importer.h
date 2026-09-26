#ifndef RESEARCH_JOINT_SHARD_STABLEHLO_IMPORTER_H_
#define RESEARCH_JOINT_SHARD_STABLEHLO_IMPORTER_H_

#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/Value.h"
#include "eggc/egraph.hpp"
#include "research/joint_shard/operation_descriptors.h"

class StableHloImporter {
 public:
  explicit StableHloImporter(eggc::EGraph& graph,
                             OperationDescriptors* descriptors = nullptr)
      : graph_(graph), descriptors_(descriptors) {}

  eggc::Id importValue(mlir::Value value);

 private:
  eggc::EGraph& graph_;
  OperationDescriptors* descriptors_;
  llvm::DenseMap<mlir::Value, eggc::Id> cache_;
};

#endif  // RESEARCH_JOINT_SHARD_STABLEHLO_IMPORTER_H_

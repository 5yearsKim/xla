#ifndef RESEARCH_JOINT_SHARD_SHARDING_PLAN_MATERIALIZER_H_
#define RESEARCH_JOINT_SHARD_SHARDING_PLAN_MATERIALIZER_H_
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "research/joint_shard/sharding/boundary_state.h"
#include "research/joint_shard/sharding/cost_model.h"

// Inlines already lowered artifacts without rerunning propagation or changing
// their collective decisions. The mesh and exact boundary remain authoritative.
class PlanMaterializer {
 public:
  PlanMaterializer(const MeshContext& mesh, llvm::ArrayRef<mlir::Type> inputs,
                   llvm::ArrayRef<mlir::Type> outputs,
                   const BoundaryState& boundary);
  mlir::ValueRange arguments() { return function_.getArguments(); }
  std::vector<mlir::Value> inlineArtifact(const std::string& artifact,
                                          mlir::ValueRange inputs);
  std::string finish(mlir::ValueRange outputs, const Cost& expected,
                     const CostModel& model);

 private:
  MeshContext mesh_;
  BoundaryState boundary_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
  mlir::func::FuncOp function_;
  mlir::OpBuilder builder_;
};
#endif

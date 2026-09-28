#ifndef RESEARCH_JOINT_SHARD_SHARDING_REGION_EVALUATOR_H_
#define RESEARCH_JOINT_SHARD_SHARDING_REGION_EVALUATOR_H_

#include <map>
#include <utility>

#include "research/joint_shard/sharding/boundary_state.h"
#include "research/joint_shard/sharding/cost_model.h"
#include "research/joint_shard/sharding/evaluation_result.h"
#include "research/joint_shard/sharding/shardy_runner.h"

namespace joint_shard {
struct Region;
struct Candidate;

// Prepared modules preserve required output constraints and are cloned for
// evaluation.
mlir::OwningOpRef<mlir::ModuleOp> prepareCandidateModule(
    const Region& region, const Candidate& candidate, const MeshContext& mesh,
    const std::map<void*, TensorSharding>& fixed_outputs = {});

class RegionEvaluator {
 public:
  RegionEvaluator(MeshContext mesh, CostModel model = CostModel{})
      : mesh_(std::move(mesh)), model_(model) {}
  EvaluationResult evaluate(mlir::ModuleOp prepared,
                            llvm::ArrayRef<TensorSharding> inputs,
                            const ShardyRunOptions& options = {}) const;

 private:
  MeshContext mesh_;
  CostModel model_;
};

class ReshardCostOracle {
 public:
  ReshardCostOracle(MeshContext mesh, CostModel model = CostModel{})
      : mesh_(std::move(mesh)), model_(model) {}
  ReshardPlan plan(const TensorSharding& from, const TensorSharding& to,
                   mlir::Type type);
  Cost estimate(const TensorSharding& from, const TensorSharding& to,
                mlir::Type type);
  size_t cacheSize() const { return cache_.size(); }

 private:
  MeshContext mesh_;
  CostModel model_;
  std::map<std::string, ReshardPlan> cache_;
};

}  // namespace joint_shard

#endif

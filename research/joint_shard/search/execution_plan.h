#ifndef RESEARCH_JOINT_SHARD_SEARCH_EXECUTION_PLAN_H_
#define RESEARCH_JOINT_SHARD_SEARCH_EXECUTION_PLAN_H_
#include <vector>

#include "research/joint_shard/search/resolved_plan.h"

namespace joint_shard {
using ExecutionPlanId = size_t;
enum class PlanKind { Region, Adapter, Composite };
struct ExecutionPlanNode {
  PlanKind kind = PlanKind::Composite;
  RegionInterface interface;
  BoundaryState boundary;
  Cost cost;
  // Region references use a summary vector index and a local implementation ID.
  size_t region = 0;
  PlanId implementation = 0;
  ReshardPlan adapter;
  std::vector<ExecutionPlanId> children;
};
// Arena IDs remain stable as new prefixes are built. Composite costs sum their
// direct children; leaf costs are charged exactly once per execution
// occurrence.
struct ExecutionPlan {
  std::vector<ExecutionPlanNode> nodes;
  ExecutionPlanId root = 0;
  ExecutionPlanId region(size_t index, const RegionSummary& summary,
                         PlanId implementation);
  ExecutionPlanId adapter(const TensorPort& port, const ReshardPlan& adapter);
  ExecutionPlanId composite(const RegionInterface& interface,
                            const BoundaryState& boundary,
                            std::vector<ExecutionPlanId> children);
  ExecutionPlanId resolved(size_t index, const RegionSummary& summary,
                           const ResolvedRegionPlan& selected);
};
std::string materializeExecutionPlan(
    const ExecutionPlan& plan, llvm::ArrayRef<const RegionSummary*> regions,
    const MeshContext& mesh, const CostModel& model);
std::string materializeExecutionPlan(const ExecutionPlan& plan,
                                     const std::vector<RegionSummary>& regions,
                                     const MeshContext& mesh,
                                     const CostModel& model);
}  // namespace joint_shard
#endif

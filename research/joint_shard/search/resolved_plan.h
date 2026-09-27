#ifndef RESEARCH_JOINT_SHARD_SEARCH_RESOLVED_PLAN_H_
#define RESEARCH_JOINT_SHARD_SEARCH_RESOLVED_PLAN_H_
#include "research/joint_shard/search/region_summary.h"
namespace joint_shard {
using ReshardPlanner = std::function<ReshardPlan(
    const TensorSharding&, const TensorSharding&, mlir::Type)>;
struct ResolvedRegionPlan {
  PlanId requested = 0, implementation = 0;
  size_t candidate_id = 0;
  BoundaryState boundary;
  Cost core_cost, adapters_cost, cost;
  std::vector<ReshardPlan> input_adapters, output_adapters;
};
ResolvedRegionPlan resolveRegionPlan(const RegionSummary& region,
                                     PlanId requested, bool use_frontier,
                                     const ReshardPlanner& oracle);
}  // namespace joint_shard
#endif

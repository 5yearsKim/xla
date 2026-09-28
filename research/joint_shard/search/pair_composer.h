#ifndef RESEARCH_JOINT_SHARD_SEARCH_PAIR_COMPOSER_H_
#define RESEARCH_JOINT_SHARD_SEARCH_PAIR_COMPOSER_H_
#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "research/joint_shard/search/resolved_plan.h"

namespace joint_shard {

struct ComposedPlan {
  BoundaryState boundary;
  ResolvedRegionPlan a, b;
  ReshardPlan intermediate;
  Cost cost;
  std::string lowered_mlir;
  std::optional<PlanId> replacement;
  std::vector<ReshardPlan> input_adapters, output_adapters;
};
struct PairSummary {
  size_t a_region = 0, b_region = 0;
  PairInterface interface;
  bool truncated = false;
  size_t evaluations = 0, shared_layout_rejections = 0,
         unknown_cost_rejections = 0;
  std::vector<ComposedPlan> plans;
  std::vector<PlanId> frontier;
  std::vector<DominanceWitness> dominance;
};
// Bounded composition with boundary deduplication and automatic dominance.
PairSummary composePair(const RegionSummary& a, const RegionSummary& b,
                        const PairInterface& interface,
                        const ReshardPlanner& oracle,
                        size_t max_evaluations = 65536);
std::string materializePair(const RegionSummary& a, const RegionSummary& b,
                            const PairSummary& pair, PlanId requested,
                            const MeshContext& mesh, const CostModel& model);
}  // namespace joint_shard

#endif

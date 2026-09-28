#include "research/joint_shard/search/region_summary.h"

#include <algorithm>
#include <stdexcept>

namespace joint_shard {

void RegionSummary::record(size_t candidate_id, const BoundaryState& boundary,
                           EvaluationResult result) {
  ++evaluations;
  if (!result.feasible) {
    ++failures[result.failure];
    return;
  }
  ++feasible_plans;
  if (!result.cost.known()) ++unknown_cost_plans;
  auto previous = std::find_if(
      plans.begin(), plans.end(),
      [&](const RegionPlan& plan) { return plan.boundary == boundary; });
  bool better = previous == plans.end();
  if (!better)
    better = result.cost.known() &&
             (!previous->cost.known() ||
              result.cost.total() < previous->cost.total() ||
              (result.cost.total() == previous->cost.total() &&
               candidate_id < previous->candidate_id));
  if (!better) return;
  RegionPlan winner{candidate_id, boundary, result.cost,
                    std::move(result.lowered_mlir)};
  if (previous == plans.end())
    plans.push_back(std::move(winner));
  else
    *previous = std::move(winner);
}
void RegionSummary::finalizePlans() {
  std::sort(plans.begin(), plans.end(),
            [](const RegionPlan& a, const RegionPlan& b) {
              return a.boundary.key() < b.boundary.key();
            });
  best_boundary_plans = plans.size();
  frontier.clear();
  dominance.clear();
  for (size_t i = 0; i < plans.size(); ++i) {
    plans[i].id = i;
    plans[i].replacement.reset();
    frontier.push_back(i);
  }
}

BoundaryPruning pruneBoundaryPlans(const RegionInterface& interface,
                                   llvm::ArrayRef<BoundaryState> boundaries,
                                   llvm::ArrayRef<Cost> costs,
                                   const ReshardEstimator& estimate) {
  if (boundaries.size() != costs.size())
    throw std::invalid_argument("boundary/cost arity mismatch");
  BoundaryPruning result;
  for (size_t i = 0; i < boundaries.size(); ++i) {
    if (boundaries[i].inputs.size() != interface.inputs.size() ||
        boundaries[i].outputs.size() != interface.outputs.size())
      throw std::invalid_argument("summary interface/type arity mismatch");
    result.frontier.push_back(i);
  }
  std::stable_sort(
      result.frontier.begin(), result.frontier.end(), [&](PlanId l, PlanId r) {
        if (costs[l].known() != costs[r].known()) return costs[l].known();
        if (costs[l].known() && costs[l].total() != costs[r].total())
          return costs[l].total() < costs[r].total();
        return boundaries[l].key() < boundaries[r].key();
      });
  std::vector<PlanId> retained;
  for (auto id : result.frontier) {
    bool dominated = false;
    if (costs[id].known())
      for (auto alternative : retained) {
        if (!costs[alternative].known()) continue;
        Cost adapters;
        for (size_t i = 0; i < interface.inputs.size(); ++i)
          adapters += estimate(boundaries[id].inputs[i],
                               boundaries[alternative].inputs[i],
                               interface.inputs[i].type);
        for (size_t i = 0; i < interface.outputs.size(); ++i)
          adapters +=
              estimate(boundaries[alternative].outputs[i],
                       boundaries[id].outputs[i], interface.outputs[i].type);
        double total = costs[alternative].total() + adapters.total();
        if (adapters.known() && total <= costs[id].total()) {
          result.dominance.push_back({boundaries[id], boundaries[alternative],
                                      adapters, total, id, alternative});
          dominated = true;
          break;
        }
      }
    if (!dominated) retained.push_back(id);
  }
  result.frontier = std::move(retained);
  return result;
}

void pruneDominatedStates(RegionSummary& summary,
                          const ReshardEstimator& estimate) {
  std::vector<BoundaryState> boundaries;
  std::vector<Cost> costs;
  for (const auto& plan : summary.plans) {
    boundaries.push_back(plan.boundary);
    costs.push_back(plan.cost);
  }
  auto pruned =
      pruneBoundaryPlans(summary.interface, boundaries, costs, estimate);
  summary.frontier = std::move(pruned.frontier);
  summary.dominance = std::move(pruned.dominance);
  for (const auto& witness : summary.dominance) {
    auto& plan = summary.plans[witness.removed_id];
    plan.replacement = witness.replacement_id;
    std::string{}.swap(plan.lowered_mlir);
  }
}

}  // namespace joint_shard

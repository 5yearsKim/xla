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
    frontier.push_back(i);
  }
}

void pruneDominatedStates(RegionSummary& summary,
                          const ReshardEstimator& estimate) {
  for (const auto& plan : summary.plans)
    if (plan.boundary.inputs.size() != summary.interface.inputs.size() ||
        plan.boundary.outputs.size() != summary.interface.outputs.size())
      throw std::invalid_argument("summary interface/type arity mismatch");
  summary.frontier.clear();
  for (size_t i = 0; i < summary.plans.size(); ++i)
    summary.frontier.push_back(i);
  std::stable_sort(
      summary.frontier.begin(), summary.frontier.end(),
      [&](PlanId l, PlanId r) {
        const auto& left = summary.plans.at(l);
        const auto& right = summary.plans.at(r);
        if (left.cost.known() != right.cost.known()) return left.cost.known();
        if (left.cost.known() && left.cost.total() != right.cost.total())
          return left.cost.total() < right.cost.total();
        return left.boundary.key() < right.boundary.key();
      });
  summary.dominance.clear();
  std::vector<PlanId> retained;
  for (auto id : summary.frontier) {
    const auto& plan = summary.plans.at(id);
    bool dominated = false;
    if (plan.cost.known())
      for (auto alternative_id : retained) {
        const auto& alternative = summary.plans.at(alternative_id);
        if (!alternative.cost.known()) continue;
        Cost adapters;
        for (size_t i = 0; i < summary.interface.inputs.size(); ++i)
          adapters +=
              estimate(plan.boundary.inputs[i], alternative.boundary.inputs[i],
                       summary.interface.inputs[i].type);
        for (size_t i = 0; i < summary.interface.outputs.size(); ++i)
          adapters += estimate(alternative.boundary.outputs[i],
                               plan.boundary.outputs[i],
                               summary.interface.outputs[i].type);
        double total = alternative.cost.total() + adapters.total();
        if (adapters.known() && total <= plan.cost.total()) {
          summary.dominance.push_back({plan.boundary, alternative.boundary,
                                       adapters, total, id, alternative_id});
          dominated = true;
          break;
        }
      }
    if (!dominated) retained.push_back(id);
  }
  summary.frontier = std::move(retained);
}

}  // namespace joint_shard

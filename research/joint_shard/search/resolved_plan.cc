#include "research/joint_shard/search/resolved_plan.h"

#include <algorithm>
#include <stdexcept>
namespace joint_shard {
ResolvedRegionPlan resolveRegionPlan(const RegionSummary& region,
                                     PlanId requested,
                                     const ReshardPlanner& oracle) {
  const auto& exact = region.plans.at(requested);
  ResolvedRegionPlan result;
  result.requested = requested;
  result.implementation = exact.replacement.value_or(requested);
  result.boundary = exact.boundary;
  if (std::find(region.frontier.begin(), region.frontier.end(),
                result.implementation) == region.frontier.end())
    throw std::invalid_argument(
        "replacement must reference a surviving implementation");
  const auto& core = region.plans.at(result.implementation);
  result.candidate_id = core.candidate_id;
  result.core_cost = core.cost;
  for (size_t i = 0; i < exact.boundary.inputs.size(); ++i) {
    auto adapter = oracle(exact.boundary.inputs[i], core.boundary.inputs[i],
                          region.interface.inputs.at(i).type);
    result.adapters_cost += adapter.cost;
    if (!adapter.feasible) ++result.adapters_cost.unknown;
    result.input_adapters.push_back(std::move(adapter));
  }
  for (size_t i = 0; i < exact.boundary.outputs.size(); ++i) {
    auto adapter = oracle(core.boundary.outputs[i], exact.boundary.outputs[i],
                          region.interface.outputs.at(i).type);
    result.adapters_cost += adapter.cost;
    if (!adapter.feasible) ++result.adapters_cost.unknown;
    result.output_adapters.push_back(std::move(adapter));
  }
  result.cost = result.core_cost;
  result.cost += result.adapters_cost;
  if (exact.cost.known() && result.cost.known() &&
      result.cost.total() > exact.cost.total() + 1e-9)
    throw std::invalid_argument(
        "resolved plan exceeds exact cost (inconsistent oracle)");
  return result;
}
}  // namespace joint_shard

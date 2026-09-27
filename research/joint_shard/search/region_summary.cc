#include "research/joint_shard/search/region_summary.h"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>

void RegionSummary::record(size_t candidate_id, const BoundaryState& boundary,
                           EvaluationResult result) {
  ++evaluations;
  if (!result.feasible) {
    ++failures[result.failure];
    return;
  }
  ++feasible_plans;
  if (!result.cost.known()) ++unknown_cost_plans;
  plans.push_back({candidate_id, boundary, result.cost,
                   result.snapshots.empty()
                       ? ""
                       : std::move(result.snapshots.back().mlir)});
}
void RegionSummary::keepBestPerBoundary() {
  std::map<std::string, RegionPlan> best;
  for (auto& plan : plans) {
    auto key = plan.boundary.key();
    auto found = best.find(key);
    bool better = found == best.end();
    if (!better) {
      const auto& previous = found->second;
      better =
          plan.cost.known() && (!previous.cost.known() ||
                                plan.cost.total() < previous.cost.total() ||
                                (plan.cost.total() == previous.cost.total() &&
                                 plan.candidate_id < previous.candidate_id));
    }
    if (better) best.insert_or_assign(std::move(key), std::move(plan));
  }
  plans.clear();
  for (auto& [key, plan] : best) plans.push_back(std::move(plan));
  best_boundary_plans = plans.size();
}

void pruneDominatedStates(RegionSummary& summary,
                          const ReshardEstimator& estimate) {
  for (const auto& plan : summary.plans)
    if (plan.boundary.inputs.size() != summary.input_types.size() ||
        plan.boundary.outputs.size() != summary.output_types.size())
      throw std::invalid_argument("summary interface/type arity mismatch");
  std::stable_sort(
      summary.plans.begin(), summary.plans.end(),
      [](const RegionPlan& left, const RegionPlan& right) {
        if (left.cost.known() != right.cost.known()) return left.cost.known();
        if (left.cost.known() && left.cost.total() != right.cost.total())
          return left.cost.total() < right.cost.total();
        return left.boundary.key() < right.boundary.key();
      });
  std::vector<RegionPlan> retained;
  for (auto& plan : summary.plans) {
    bool dominated = false;
    if (plan.cost.known())
      for (const auto& alternative : retained) {
        if (!alternative.cost.known()) continue;
        Cost adapters;
        for (size_t i = 0; i < summary.input_types.size(); ++i)
          adapters +=
              estimate(plan.boundary.inputs[i], alternative.boundary.inputs[i],
                       summary.input_types[i]);
        for (size_t i = 0; i < summary.output_types.size(); ++i)
          adapters +=
              estimate(alternative.boundary.outputs[i],
                       plan.boundary.outputs[i], summary.output_types[i]);
        double total = alternative.cost.total() + adapters.total();
        if (adapters.known() && total <= plan.cost.total()) {
          summary.dominance.push_back(
              {plan.boundary, alternative.boundary, adapters, total});
          dominated = true;
          break;
        }
      }
    if (!dominated) retained.push_back(std::move(plan));
  }
  summary.plans = std::move(retained);
}

std::string RegionSummary::str() const {
  std::ostringstream out;
  out << "Region " << id << "\n========\nOperations: " << operations
      << "\nInputs: " << inputs << "\nOutputs: " << outputs
      << "\nCandidates extracted: " << candidates.size()
      << "\nBoundary states evaluated: " << boundaries_evaluated
      << "\nEvaluations: " << evaluations
      << "\nBefore best-per-boundary: " << feasible_plans << " plans"
      << "\nAfter best-per-boundary: " << best_boundary_plans << " plans"
      << "\nAfter reshard dominance: " << plans.size() << " plans"
      << "\nUnknown-cost evaluations: " << unknown_cost_plans
      << "\nOversized protected region: " << oversized
      << "\nBoundary search truncated: " << boundary_search_truncated << '\n';
  for (const auto& [reason, count] : failures)
    out << "Rejected: " << reason << " count=" << count << '\n';
  for (const auto& candidate : candidates)
    out << "Candidate R" << candidate.id << ": " << candidate.name << '\n';
  out << "\nBoundary | Rewrite | Compute(us) | Comm(us) | Total(us)\n";
  out << std::setprecision(12);
  for (const auto& plan : plans) {
    out << plan.boundary.key() << " | R" << plan.candidate_id << " | ";
    if (plan.cost.known())
      out << plan.cost.compute << " | " << plan.cost.communication << " | "
          << plan.cost.total();
    else
      out << "unknown | unknown | unknown";
    out << '\n';
  }
  for (const auto& witness : dominance)
    out << "Pruned " << witness.removed.key() << " via "
        << witness.replacement.key()
        << " adapters_us=" << witness.adapters.total()
        << " replacement_total_us=" << witness.replacement_total << '\n';
  return out.str();
}

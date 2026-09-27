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

std::string RegionSummary::str() const {
  std::ostringstream out;
  out << "Region " << id << "\n========\nOperations: " << operations
      << "\nInputs: " << interface.inputs.size()
      << "\nOutputs: " << interface.outputs.size()
      << "\nCandidates extracted: " << candidates.size()
      << "\nBoundary states evaluated: " << boundaries_evaluated
      << "\nEvaluations: " << evaluations
      << "\nBefore best-per-boundary: " << feasible_plans << " plans"
      << "\nAfter best-per-boundary: " << best_boundary_plans << " plans"
      << "\nAfter reshard dominance: " << frontier.size() << " plans"
      << "\nUnknown-cost evaluations: " << unknown_cost_plans
      << "\nOversized protected region: " << oversized
      << "\nBoundary search truncated: " << boundary_search_truncated << '\n';
  for (const auto& [reason, count] : failures)
    out << "Rejected: " << reason << " count=" << count << '\n';
  for (const auto& candidate : candidates)
    out << "Candidate R" << candidate.id << ": " << candidate.name << '\n';
  out << "\nBoundary | Rewrite | Compute(us) | Comm(us) | Total(us)\n";
  out << std::setprecision(12);
  for (auto id : frontier) {
    const auto& plan = plans.at(id);
    out << "P" << id << " " << plan.boundary.key() << " | R"
        << plan.candidate_id << " | ";
    if (plan.cost.known())
      out << plan.cost.compute << " | " << plan.cost.communication << " | "
          << plan.cost.total();
    else
      out << "unknown | unknown | unknown";
    out << '\n';
  }
  for (const auto& witness : dominance)
    out << "Pruned P" << witness.removed_id << " " << witness.removed.key()
        << " via P" << witness.replacement_id << " "
        << witness.replacement.key()
        << " adapters_us=" << witness.adapters.total()
        << " replacement_total_us=" << witness.replacement_total << '\n';
  return out.str();
}

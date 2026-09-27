#ifndef RESEARCH_JOINT_SHARD_SEARCH_REGION_SUMMARY_H_
#define RESEARCH_JOINT_SHARD_SEARCH_REGION_SUMMARY_H_

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "research/joint_shard/search/region_interface.h"
#include "research/joint_shard/sharding/region_evaluator.h"
using PlanId = size_t;

struct RegionPlan {
  size_t candidate_id = 0;
  BoundaryState boundary;
  Cost cost;
  std::string lowered_mlir;
  PlanId id = 0;
};
struct DominanceWitness {
  BoundaryState removed;
  BoundaryState replacement;
  Cost adapters;
  double replacement_total = 0;
  PlanId removed_id = 0, replacement_id = 0;
};
struct RegionSummary {
  size_t id = 0;
  size_t operations = 0;
  bool oversized = false;
  bool boundary_search_truncated = false;
  size_t boundaries_evaluated = 0;
  size_t evaluations = 0;
  size_t feasible_plans = 0;
  size_t unknown_cost_plans = 0;
  size_t best_boundary_plans = 0;
  std::map<std::string, size_t> failures;
  std::vector<Candidate> candidates;
  RegionInterface interface;
  // Complete exact table; pruning only modifies frontier and witnesses.
  std::vector<RegionPlan> plans;
  std::vector<PlanId> frontier;
  std::vector<DominanceWitness> dominance;

  void record(size_t candidate_id, const BoundaryState& boundary,
              EvaluationResult result);
  void keepBestPerBoundary();
  std::string str() const;
};
using ReshardEstimator = std::function<Cost(const TensorSharding&,
                                            const TensorSharding&, mlir::Type)>;
// Conservative O(n^2) pruning against retained plans only. Every witness points
// directly to a surviving plan; equal-cost states cannot delete one another.
void pruneDominatedStates(RegionSummary& summary,
                          const ReshardEstimator& estimate);

#endif

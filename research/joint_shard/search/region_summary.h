#ifndef RESEARCH_JOINT_SHARD_SEARCH_REGION_SUMMARY_H_
#define RESEARCH_JOINT_SHARD_SEARCH_REGION_SUMMARY_H_

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "research/joint_shard/sharding/region_evaluator.h"

struct RegionPlan {
  size_t candidate_id = 0;
  BoundaryState boundary;
  Cost cost;
  std::string lowered_mlir;
};
struct DominanceWitness {
  BoundaryState removed;
  BoundaryState replacement;
  Cost adapters;
  double replacement_total = 0;
};
struct RegionSummary {
  size_t id = 0;
  size_t operations = 0;
  size_t inputs = 0;
  size_t outputs = 0;
  bool oversized = false;
  bool boundary_search_truncated = false;
  size_t boundaries_evaluated = 0;
  size_t evaluations = 0;
  size_t feasible_plans = 0;
  size_t unknown_cost_plans = 0;
  size_t best_boundary_plans = 0;
  std::map<std::string, size_t> failures;
  std::vector<mlir::Type> input_types;
  std::vector<mlir::Type> output_types;
  std::vector<Candidate> candidates;
  std::vector<RegionPlan> plans;
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

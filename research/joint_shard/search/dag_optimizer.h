#ifndef RESEARCH_JOINT_SHARD_SEARCH_DAG_OPTIMIZER_H_
#define RESEARCH_JOINT_SHARD_SEARCH_DAG_OPTIMIZER_H_
#include "research/joint_shard/search/chain_optimizer.h"

namespace joint_shard {
struct DagSearchOptions {
  size_t max_live_values = 4;
  size_t max_states = 4096;
  size_t max_transitions = 65536;
};
struct DagStep {
  size_t region = 0;
  ResolvedRegionPlan selected;
  // One per distinct region input port, including zero-cost identity adapters.
  std::vector<ReshardPlan> incoming;
  std::vector<ReshardPlan> outgoing;
  // Layouts in DagInterface.live_after[region] order.
  std::vector<TensorSharding> live_layouts;
};
struct DagResult {
  BoundaryState contract;
  bool feasible = false, truncated = false;
  size_t transitions = 0, states_retained = 0, states_discarded = 0,
         peak_live_values = 0, contract_rejections = 0,
         unknown_cost_rejections = 0;
  std::vector<size_t> states_per_layer, discarded_per_layer;
  std::vector<bool> transitions_truncated_per_layer;
  size_t dominance_pruned = 0;
  double search_us = 0;
  std::string failure;
  Cost cost;
  std::vector<DagStep> steps;
  ExecutionPlan execution;
  std::string lowered_mlir;
};
// Fixed source region order, one producer-layout copy per live value, and
// consumer-local conversions. Prefix dominance preserves contracts through
// costed output adapters; budgets bound active states and attempted
// transitions.
DagResult optimizeDag(const std::vector<RegionSummary>& regions,
                      const DagInterface& interface,
                      const BoundaryState& contract,
                      const ReshardPlanner& oracle,
                      const DagSearchOptions& options = {});
struct DagExperiment : FunctionSearchStatistics {
  DagInterface interface;
  DagResult result;
  const DagResult& selected() const { return result; }
};
}  // namespace joint_shard
#endif

#ifndef RESEARCH_JOINT_SHARD_SEARCH_DAG_OPTIMIZER_H_
#define RESEARCH_JOINT_SHARD_SEARCH_DAG_OPTIMIZER_H_
#include "research/joint_shard/search/chain_optimizer.h"

namespace joint_shard {
using DagSearchMode = ChainSearchMode;
struct DagSearchOptions {
  DagSearchMode mode = DagSearchMode::Exact;
  size_t max_live_values = 4;
  size_t max_states = 4096;
  size_t max_transitions = 65536;
};
struct DagStep {
  size_t region = 0;
  ResolvedRegionPlan selected;
  // One per distinct region input port, including zero-cost identity adapters.
  std::vector<ReshardPlan> incoming;
  // Layouts in DagInterface.live_after[region] order.
  std::vector<TensorSharding> live_layouts;
};
struct DagResult {
  DagSearchMode mode = DagSearchMode::Exact;
  BoundaryState contract;
  bool feasible = false, truncated = false;
  size_t transitions = 0, states_retained = 0, states_discarded = 0,
         peak_live_values = 0, contract_rejections = 0,
         unknown_cost_rejections = 0;
  std::vector<size_t> states_per_layer, discarded_per_layer;
  std::vector<bool> transitions_truncated_per_layer;
  double search_us = 0;
  std::string failure;
  Cost cost;
  std::vector<DagStep> steps;
  ExecutionPlan execution;
  std::string lowered_mlir;
};
// Fixed source region order, one producer-layout copy per live value, and
// consumer-local conversions. Uncapped exact search minimizes additive cost
// over the supplied region tables under this model.
DagResult optimizeDag(const std::vector<RegionSummary>& regions,
                      const DagInterface& interface,
                      const BoundaryState& contract,
                      const ReshardPlanner& oracle,
                      const DagSearchOptions& options = {});
struct DagExperiment : FunctionSearchStatistics {
  DagInterface interface;
  // Original uses OptimizationReport.original_regions; joint modes use regions.
  DagResult original, greedy, exact, resolved;
  bool select_resolved = false;
  const DagResult& selected() const {
    return select_resolved ? resolved : exact;
  }
};
}  // namespace joint_shard
#endif

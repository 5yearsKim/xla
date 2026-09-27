#ifndef RESEARCH_JOINT_SHARD_SEARCH_CHAIN_OPTIMIZER_H_
#define RESEARCH_JOINT_SHARD_SEARCH_CHAIN_OPTIMIZER_H_
#include <optional>

#include "research/joint_shard/search/execution_plan.h"
namespace joint_shard {
enum class ChainSearchMode { Exact, Resolved, Greedy };
struct ChainSearchOptions {
  ChainSearchMode mode = ChainSearchMode::Exact;
  // Per layer: a bounded layer still supplies prefixes to subsequent layers.
  size_t max_transitions = 65536;
};
struct ChainStep {
  size_t region = 0;
  ResolvedRegionPlan selected;
  // First region has no incoming adapter.
  std::optional<ReshardPlan> incoming;
};
struct ChainResult {
  ChainSearchMode mode = ChainSearchMode::Exact;
  BoundaryState contract;
  bool feasible = false, truncated = false;
  size_t transitions = 0, states_retained = 0, contract_rejections = 0,
         unknown_cost_rejections = 0;
  double search_us = 0;
  std::string failure;
  Cost cost;
  std::vector<size_t> states_per_layer;
  std::vector<ChainStep> steps;
  ExecutionPlan execution;
  std::string lowered_mlir;
};
ChainResult optimizeChain(const std::vector<RegionSummary>& regions,
                          const ChainInterface& interface,
                          const BoundaryState& contract,
                          const ReshardPlanner& oracle,
                          const ChainSearchOptions& options = {});
struct FunctionSearchStatistics {
  BoundaryState contract;
  bool boundary_truncated = false, candidate_cap_reached = false,
       saturation_limited = false, extraction_limited = false;
  size_t extraction_states = 0;
  double extraction_us = 0;
  std::string numerical_policy;
  double evaluation_us = 0;
};
struct ChainExperiment : FunctionSearchStatistics {
  ChainInterface interface;
  // Original refers to OptimizationReport.original_regions; the other plans
  // refer to OptimizationReport.regions.
  ChainResult original, greedy, exact, resolved;
  bool select_resolved = false;
  const ChainResult& selected() const {
    return select_resolved ? resolved : exact;
  }
};
}  // namespace joint_shard
#endif

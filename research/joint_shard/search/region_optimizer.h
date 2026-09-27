#ifndef RESEARCH_JOINT_SHARD_SEARCH_REGION_OPTIMIZER_H_
#define RESEARCH_JOINT_SHARD_SEARCH_REGION_OPTIMIZER_H_

#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "eggc/runner.hpp"
#include "research/joint_shard/search/chain_optimizer.h"
#include "research/joint_shard/search/dag_optimizer.h"
#include "research/joint_shard/search/optimization_observer.h"
#include "research/joint_shard/search/pair_composer.h"
#include "research/joint_shard/sharding/boundary_state.h"
#include "research/joint_shard/sharding/cost_model.h"
#include "research/joint_shard/transforms/regionizer.h"
#include "research/joint_shard/transforms/rewrite_options.h"

namespace joint_shard {

struct RegionOptimizerOptions {
  RegionizerOptions regionizer;
  TensorRewriteOptions rewriting;
  LayoutPolicy layouts;
  CostModelOptions cost;
  std::optional<std::pair<size_t, size_t>> compose_regions;
  size_t max_pair_evaluations = 65536;
  bool compose_pruned = false;
  bool optimize_chain = false;
  bool chain_resolved = false;
  size_t max_chain_transitions = 65536;
  bool optimize_dag = false;
  DagSearchOptions dag;
  size_t max_candidates = 32;
  size_t max_boundary_states = 256;
  std::string mesh_name;
};
struct OptimizationReport {
  MeshContext mesh;
  std::vector<RegionSummary> regions;
  // Original-only implementations retained for function baseline references.
  std::vector<RegionSummary> original_regions;
  std::vector<PairSummary> compositions;
  std::optional<ChainExperiment> chain;
  std::optional<DagExperiment> dag;
  std::map<std::string, size_t> preserved_operations;
  std::vector<eggc::RunReport> saturation;
};
// Does not mutate the source module. Optional composition produces standalone
// IR.
OptimizationReport summarizeRegions(mlir::ModuleOp module,
                                    const RegionOptimizerOptions& options = {},
                                    const OptimizationObserver& observer = {});

}  // namespace joint_shard

#endif

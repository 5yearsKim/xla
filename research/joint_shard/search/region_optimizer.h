#ifndef RESEARCH_JOINT_SHARD_SEARCH_REGION_OPTIMIZER_H_
#define RESEARCH_JOINT_SHARD_SEARCH_REGION_OPTIMIZER_H_

#include "research/joint_shard/search/region_summary.h"

struct RegionOptimizerOptions {
  RegionizerOptions regionizer;
  TensorRewriteOptions rewriting;
  LayoutPolicy layouts;
  CostModelOptions cost;
  size_t max_candidates = 32;
  size_t max_boundary_states = 256;
  std::string mesh_name;
  std::string dump_directory;
};
struct OptimizationReport {
  MeshContext mesh;
  std::vector<RegionSummary> regions;
  std::map<std::string, size_t> preserved_operations;
  std::vector<eggc::RunReport> saturation;
  std::string str() const;
};
// Does not mutate the source module or compose neighboring regions.
OptimizationReport summarizeRegions(mlir::ModuleOp module,
                                    const RegionOptimizerOptions& options = {});

#endif

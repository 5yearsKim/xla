#ifndef RESEARCH_JOINT_SHARD_TRANSFORMS_REWRITE_REGIONS_H_
#define RESEARCH_JOINT_SHARD_TRANSFORMS_REWRITE_REGIONS_H_

#include <map>
#include <string>
#include <vector>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "eggc/runner.hpp"
#include "research/joint_shard/bridge/tensor_lang/tensor_rewrites.h"
#include "research/joint_shard/transforms/rewrite_options.h"

namespace joint_shard {

struct TensorRewriteReport {
  std::size_t regions = 0;
  std::size_t imported_operations = 0;
  std::size_t roots = 0;
  std::map<std::string, std::size_t> boundaries;
  std::map<std::string, SemanticRuleStats> rules;
  std::vector<eggc::RunReport> runs;
  std::vector<TensorExtractionReport> extractions;
};
// Supported contiguous islands use one graph and one export cache per island.
// Fixed shardings, constraints, unknown metadata, and unsupported regions are
// preserved as boundaries. All outputs of an island share extracted values.
mlir::LogicalResult rewriteUnconstrainedRegions(
    mlir::ModuleOp module, const TensorRewriteOptions& options = {},
    TensorRewriteReport* report = nullptr);
}  // namespace joint_shard

#endif

#ifndef RESEARCH_JOINT_SHARD_TRANSFORMS_REWRITE_REGIONS_H_
#define RESEARCH_JOINT_SHARD_TRANSFORMS_REWRITE_REGIONS_H_

#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "eggc/runner.hpp"
#include "research/joint_shard/bridge/tensor_lang/tensor_rewrites.h"
#include "research/joint_shard/transforms/tensor_extraction.h"

struct TensorRewriteOptions {
  std::string rules_file;  // Empty uses the embedded tensor.rules.
  NumericalPolicy numerical_policy = NumericalPolicy::PreserveEvaluation;
  eggc::RunOptions runner = [] {
    eggc::RunOptions limits;
    limits.match_limit = 4096;
    limits.per_rule_match_limit = 256;
    limits.time_limit = std::chrono::milliseconds(1000);
    return limits;
  }();
  SemanticRuleOptions semantic;
  ExtractionProfile extraction = ExtractionProfile::Compute;
  TensorExtractorMode extractor = TensorExtractorMode::Auto;
  eggc::DagOptions dag = [] {
    eggc::DagOptions limits;
    limits.state_limit = 10000;
    limits.time_limit = std::chrono::milliseconds(50);
    limits.frontier_limit = 1000;
    return limits;
  }();
};
struct TensorRewriteReport {
  std::size_t regions = 0;
  std::size_t imported_operations = 0;
  std::size_t roots = 0;
  std::map<std::string, std::size_t> boundaries;
  std::map<std::string, SemanticRuleStats> rules;
  std::vector<eggc::RunReport> runs;
  std::vector<TensorExtractionReport> extractions;
  std::string str() const;
};
// Supported contiguous islands use one graph and one export cache per island.
// Fixed shardings, constraints, unknown metadata, and unsupported regions are
// preserved as boundaries. All outputs of an island share extracted values.
mlir::LogicalResult rewriteUnconstrainedRegions(
    mlir::ModuleOp module, const TensorRewriteOptions& options = {},
    TensorRewriteReport* report = nullptr);
// Shared by both command-line tools. Returns false for non-rewrite options.
bool parseTensorRewriteOption(std::string_view argument,
                              TensorRewriteOptions& options);
std::string_view tensorRewriteOptionHelp();
#endif

#ifndef RESEARCH_JOINT_SHARD_TRANSFORMS_REWRITE_OPTIONS_H_
#define RESEARCH_JOINT_SHARD_TRANSFORMS_REWRITE_OPTIONS_H_
#include <string>
#include <string_view>

#include "eggc/runner.hpp"
#include "research/joint_shard/bridge/tensor_lang/tensor_rewrites.h"
#include "research/joint_shard/transforms/tensor_extraction.h"

struct TensorRewriteOptions {
  std::string rules_file;     // Empty uses the embedded tensor.rules.
  bool print_egraph = false;  // Print post-rewrite e-classes to stderr.
  NumericalPolicy numerical_policy = NumericalPolicy::AllowReassociation;
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
bool parseTensorRewriteOption(std::string_view argument,
                              TensorRewriteOptions& options);
std::string_view tensorRewriteOptionHelp();
#endif

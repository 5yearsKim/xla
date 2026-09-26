#ifndef RESEARCH_JOINT_SHARD_BRIDGE_TENSOR_LANG_TENSOR_REWRITES_H_
#define RESEARCH_JOINT_SHARD_BRIDGE_TENSOR_LANG_TENSOR_REWRITES_H_
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "eggc/rewrite.hpp"
#include "research/joint_shard/bridge/tensor_lang/op_properties.h"
#include "research/joint_shard/bridge/tensor_lang/tensor_analysis.h"

using TensorCondition = eggc::Condition<TensorNode, TensorAnalysis>;
using TensorRewrite = eggc::Rewrite<TensorNode, TensorAnalysis>;
using TensorSubstitution = eggc::Substitution;

TensorCondition sameTensorType(std::string lhs, std::string rhs);
// Ordinary patterns match complete operator attrs; guards inspect tensor facts.
std::vector<TensorRewrite> buildTensorRewrites(
    NumericalPolicy policy = NumericalPolicy::PreserveEvaluation);

struct SemanticRuleStats {
  std::size_t node_visits = 0;
  std::size_t structural_matches = 0;
  std::size_t accepted = 0;
  std::size_t applied = 0;
  std::size_t duplicates = 0;
  std::size_t budget_stops = 0;
  std::map<std::string, std::size_t> rejections;
};
struct SemanticRuleOptions {
  std::size_t visit_limit = 100000;
  std::size_t match_limit = 10000;
  std::shared_ptr<std::map<std::string, SemanticRuleStats>> report;
  std::string source_name = "<semantic rules>";
  std::optional<NumericalPermissions> permissions;
  bool enable_associativity = true;
  bool enable_linearity = true;
  bool enable_homogeneity = true;
};
std::vector<TensorRewrite> buildAttributeRewrites();
std::string_view defaultSemanticRules();

// Parse semantic rewrite declarations and lower them to TensorLang custom
// search rules. Operator predicates use the shared OpProperty registry;
// Scalar(?v)/Uniform(?v) query value facts. scale(?s, ?x) matches scalar
// broadcast multiplication and constructs a broadcast at the RHS tensor shape.
std::vector<TensorRewrite> parseSemanticRules(
    std::string_view source,
    NumericalPolicy policy = NumericalPolicy::PreserveEvaluation,
    SemanticRuleOptions options = {});
std::vector<TensorRewrite> loadSemanticRulesFile(
    std::string_view path,
    NumericalPolicy policy = NumericalPolicy::PreserveEvaluation,
    SemanticRuleOptions options = {});
#endif  // RESEARCH_JOINT_SHARD_BRIDGE_TENSOR_LANG_TENSOR_REWRITES_H_

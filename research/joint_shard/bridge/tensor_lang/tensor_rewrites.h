#ifndef RESEARCH_JOINT_SHARD_BRIDGE_TENSOR_LANG_TENSOR_REWRITES_H_
#define RESEARCH_JOINT_SHARD_BRIDGE_TENSOR_LANG_TENSOR_REWRITES_H_
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "eggc/rewrite.hpp"
#include "research/joint_shard/bridge/tensor_lang/op_properties.h"
#include "research/joint_shard/bridge/tensor_lang/tensor_analysis.h"

namespace joint_shard {

using TensorRewrite = eggc::Rewrite<TensorNode, TensorAnalysis>;

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
  std::optional<NumericalPermissions> permissions;
  bool enable_associativity = true;
  bool enable_linearity = true;
  bool enable_homogeneity = true;
};
std::vector<TensorRewrite> buildAttributeRewrites();
namespace patterns {
class RuleDefinition;
}
// Compile trusted C++ pattern definitions to bounded, checked rewrite rules.
std::vector<TensorRewrite> compileRules(
    std::vector<patterns::RuleDefinition> definitions,
    NumericalPolicy policy = NumericalPolicy::PreserveEvaluation,
    SemanticRuleOptions options = {});
std::vector<TensorRewrite> buildSemanticRules(
    NumericalPolicy policy = NumericalPolicy::PreserveEvaluation,
    SemanticRuleOptions options = {});
}  // namespace joint_shard

#endif  // RESEARCH_JOINT_SHARD_BRIDGE_TENSOR_LANG_TENSOR_REWRITES_H_

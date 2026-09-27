#include <fstream>
#include <sstream>
#include <stdexcept>

#include "research/joint_shard/bridge/tensor_lang/semantic_rule_internal.h"

namespace joint_shard {
std::vector<TensorRewrite> parseSemanticRules(std::string_view source,
                                              NumericalPolicy policy,
                                              SemanticRuleOptions options) {
  auto parsed = semantic_detail::parseRules(source, options.source_name);
  std::vector<TensorRewrite> result;
  for (auto& rule : parsed) {
    bool enabled = true;
    for (const auto& predicate : rule.predicates) {
      if (predicate.name == "Associative" && !options.enable_associativity)
        enabled = false;
      if (predicate.name == "LinearIn" && !options.enable_linearity)
        enabled = false;
      if (predicate.name == "HomogeneousIn" && !options.enable_homogeneity)
        enabled = false;
    }
    if (enabled)
      result.push_back(
          semantic_detail::lowerRule(std::move(rule), policy, options));
  }
  return result;
}
std::vector<TensorRewrite> loadSemanticRulesFile(std::string_view path,
                                                 NumericalPolicy policy,
                                                 SemanticRuleOptions options) {
  std::ifstream input{std::string(path)};
  if (!input)
    throw std::runtime_error("cannot open semantic rules file: " +
                             std::string(path));
  std::ostringstream contents;
  contents << input.rdbuf();
  if (!input.good() && !input.eof())
    throw std::runtime_error("failed reading semantic rules file: " +
                             std::string(path));
  options.source_name = std::string(path);
  return parseSemanticRules(contents.str(), policy, std::move(options));
}

std::string_view defaultSemanticRules() {
  static constexpr std::string_view rules =
#include "research/joint_shard/bridge/tensor_lang/tensor_rules.inc"
      ;
  return rules;
}

}  // namespace joint_shard

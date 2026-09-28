#include <stdexcept>
#include <unordered_set>

#include "research/joint_shard/bridge/tensor_lang/semantic_rule_internal.h"

namespace joint_shard {
std::vector<TensorRewrite> compileRules(
    std::vector<patterns::RuleDefinition> definitions, NumericalPolicy policy,
    SemanticRuleOptions options) {
  std::vector<semantic_detail::Rule> validated;
  std::unordered_set<std::string> names;
  for (const auto& definition : definitions) {
    auto rule = semantic_detail::PatternAccess::definition(definition);
    semantic_detail::validateRule(rule);
    if (!names.insert(rule.name).second)
      throw std::invalid_argument("duplicate rule name: " + rule.name);
    validated.push_back(std::move(rule));
  }
  std::vector<TensorRewrite> result;
  for (auto& rule : validated) {
    bool enabled = true;
    for (const auto& predicate : rule.predicates) {
      if ((predicate.kind == semantic_detail::PredicateKind::Associative ||
           predicate.kind ==
               semantic_detail::PredicateKind::DotReassociation) &&
          !options.enable_associativity)
        enabled = false;
      if ((predicate.kind == semantic_detail::PredicateKind::LinearIn ||
           predicate.kind ==
               semantic_detail::PredicateKind::SumDotInterchange ||
           predicate.kind == semantic_detail::PredicateKind::DotArithmetic ||
           predicate.kind == semantic_detail::PredicateKind::RawMoments) &&
          !options.enable_linearity)
        enabled = false;
      if (predicate.kind == semantic_detail::PredicateKind::HomogeneousIn &&
          !options.enable_homogeneity)
        enabled = false;
      if (predicate.kind == semantic_detail::PredicateKind::DotDivision &&
          !options.enable_homogeneity)
        enabled = false;
    }
    if (enabled)
      result.push_back(
          semantic_detail::lowerRule(std::move(rule), policy, options));
  }
  return result;
}

}  // namespace joint_shard

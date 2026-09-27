#ifndef RESEARCH_JOINT_SHARD_BRIDGE_TENSOR_LANG_SEMANTIC_RULE_INTERNAL_H_
#define RESEARCH_JOINT_SHARD_BRIDGE_TENSOR_LANG_SEMANTIC_RULE_INTERNAL_H_
#include <functional>
#include <unordered_map>
#include <unordered_set>

#include "research/joint_shard/bridge/tensor_lang/tensor_rewrites.h"

namespace joint_shard {
namespace semantic_detail {
struct Term {
  std::string name;
  bool variable = false;
  std::vector<Term> children;
};
struct Predicate {
  std::string name;
  std::string subject;
  std::optional<unsigned> operand;
};
struct Rule {
  std::string name;
  Term lhs;
  Term rhs;
  std::vector<Predicate> predicates;
  std::size_t line = 1;
};

using TensorBindings = std::unordered_map<std::string, eggc::Id>;
struct OperatorOccurrence {
  TensorNode node;
  eggc::Id eclass;
};
struct Bindings {
  TensorBindings tensors;
  std::unordered_map<std::string, std::vector<OperatorOccurrence>> operators;
};
struct SearchState {
  const eggc::StopCheck& stop;
  const SemanticRuleOptions& options;
  SemanticRuleStats& stats;
  NumericalPolicy policy;
  std::size_t visits = 0;
  std::size_t matches = 0;
  bool budget = false;
  bool cancelled() {
    if (stop && stop()) return true;
    if (visits >= options.visit_limit || matches >= options.match_limit) {
      if (!budget) ++stats.budget_stops;
      budget = true;
      return true;
    }
    return false;
  }
};
struct Prepared {
  std::optional<eggc::Id> reference;
  TensorNode node{};
  TensorFacts facts;
  std::vector<Prepared> children;
};
using BindingSink = std::function<bool(const Bindings&)>;
std::vector<Rule> parseRules(std::string_view source, std::string name);
bool matchTerm(const Term& term, eggc::Id id, const TensorEGraph& graph,
               const Bindings& initial, SearchState& state,
               const BindingSink& sink);
bool isUniformTensor(const TensorEGraph& graph, eggc::Id id,
                     std::unordered_set<eggc::Id>& seen, SearchState& state);
OpProperty toProperty(const Predicate& predicate);
PropertyDecision checkNode(const TensorNode& node, TensorFacts result,
                           const std::vector<TensorFacts>& operands,
                           const Predicate& predicate, NumericalPolicy policy,
                           std::optional<NumericalPermissions> permissions);
std::optional<Prepared> prepare(const Term& term, const Bindings& bindings,
                                const TensorEGraph& graph, const Rule& rule,
                                NumericalPolicy policy,
                                const SemanticRuleOptions& options,
                                std::string& rejection);
eggc::Id instantiate(const Prepared& prepared, TensorEGraph& graph);
std::string bindingKey(eggc::Id root, const Bindings& bindings);
TensorRewrite lowerRule(Rule rule, NumericalPolicy policy,
                        SemanticRuleOptions options);
}  // namespace semantic_detail
}  // namespace joint_shard

#endif

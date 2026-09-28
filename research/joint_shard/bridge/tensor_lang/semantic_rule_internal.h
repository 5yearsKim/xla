#ifndef RESEARCH_JOINT_SHARD_BRIDGE_TENSOR_LANG_SEMANTIC_RULE_INTERNAL_H_
#define RESEARCH_JOINT_SHARD_BRIDGE_TENSOR_LANG_SEMANTIC_RULE_INTERNAL_H_
#include <functional>
#include <unordered_map>
#include <unordered_set>

#include "research/joint_shard/bridge/tensor_lang/tensor_patterns.h"

namespace joint_shard {
namespace semantic_detail {
enum class TermKind {
  TensorVariable,
  OperatorVariable,
  ConcreteOperator,
  Scale
};
using AttributeValue = patterns::AttributeValue;
struct Term {
  std::string name;
  TermKind kind = TermKind::ConcreteOperator;
  OpKind op = OpKind::Input;
  std::vector<Term> children;
  OpAttrs initial_attributes = NoAttrs{};
  std::vector<patterns::AttributeConstraint> attributes;
};
enum class PredicateKind {
  Elementwise,
  Commutative,
  Associative,
  Involution,
  LinearIn,
  HomogeneousIn,
  Scalar,
  Uniform,
  Rank,
  DotReassociation,
  SumDotInterchange,
  DotArithmetic,
  DotDivision,
  RawMoments,
};
struct Predicate {
  PredicateKind kind;
  std::string subject;
  std::optional<unsigned> operand;
  bool isProperty() const { return kind <= PredicateKind::HomogeneousIn; }
  bool isRuleGuard() const {
    return kind == PredicateKind::DotReassociation ||
           kind == PredicateKind::SumDotInterchange ||
           kind == PredicateKind::DotArithmetic ||
           kind == PredicateKind::DotDivision ||
           kind == PredicateKind::RawMoments;
  }
};
struct Rule {
  std::string name;
  Term lhs;
  std::optional<Term> rhs;
  patterns::RhsCallback builder;
  std::vector<Predicate> predicates;
};

using TensorBindings = std::unordered_map<std::string, eggc::Id>;
struct OperatorOccurrence {
  TensorNode node;
  eggc::Id eclass;
};
struct Bindings {
  TensorBindings tensors;
  std::unordered_map<std::string, std::vector<OperatorOccurrence>> operators;
  std::unordered_map<std::string, AttributeValue> attributes;
  std::vector<OperatorOccurrence> concrete;
};
bool matchAttributes(const Term& term, const TensorNode& node,
                     Bindings& bindings);
std::optional<TensorNode> buildConcreteNode(
    const Term& term, const Bindings& bindings,
    std::span<const TensorFacts> operands, std::string& rejection);
std::string formatAttribute(const AttributeValue& value);
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
  std::string operator_variable;
  std::vector<Prepared> children;
};
PropertyDecision checkRuleGuard(const Predicate& predicate,
                                const Bindings& bindings, const Prepared& rhs,
                                const TensorEGraph& graph,
                                NumericalPolicy policy,
                                const SemanticRuleOptions& options);
struct PatternAccess {
  static const Term& term(const patterns::Pattern& p) { return *p.term_; }
  static patterns::Pattern pattern(Term term) {
    return patterns::Pattern(std::make_shared<const Term>(std::move(term)));
  }
  static const Predicate& predicate(const patterns::Guard& p) {
    return *p.predicate_;
  }
  static patterns::Guard guard(Predicate p) {
    return patterns::Guard(std::make_shared<const Predicate>(std::move(p)));
  }
  static const Rule& definition(const patterns::RuleDefinition& r) {
    return *r.rule_;
  }
  static patterns::RuleDefinition definition(Rule rule) {
    return patterns::RuleDefinition(
        std::make_shared<const Rule>(std::move(rule)));
  }
  static patterns::Match match(const Bindings& bindings,
                               const TensorEGraph& graph) {
    return patterns::Match(bindings, graph);
  }
  static patterns::RhsBuilder builder(const Bindings& bindings,
                                      const TensorEGraph& graph) {
    return patterns::RhsBuilder(bindings, graph);
  }
  static std::optional<Prepared> prepared(
      const patterns::Expression& expression, std::string& rejection) {
    if (!expression.prepared_) {
      rejection = expression.error_;
      return std::nullopt;
    }
    return *expression.prepared_;
  }
};
using BindingSink = std::function<bool(const Bindings&)>;
void validateRule(const Rule& rule);
bool matchTerm(const Term& term, eggc::Id id, const TensorEGraph& graph,
               const Bindings& initial, SearchState& state,
               const BindingSink& sink);
bool isUniformTensor(const TensorEGraph& graph, eggc::Id id,
                     std::unordered_set<eggc::Id>& seen, SearchState& state);
std::string bindingKey(eggc::Id root, const Bindings& bindings);
TensorRewrite lowerRule(Rule rule, NumericalPolicy policy,
                        SemanticRuleOptions options);
}  // namespace semantic_detail
}  // namespace joint_shard

#endif

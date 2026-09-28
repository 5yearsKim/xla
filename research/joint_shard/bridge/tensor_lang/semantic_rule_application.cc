#include <algorithm>
#include <utility>

#include "research/joint_shard/bridge/tensor_lang/semantic_rule_internal.h"

namespace joint_shard {
namespace semantic_detail {
namespace {
OpProperty toProperty(const Predicate& predicate) {
  switch (predicate.kind) {
    case PredicateKind::Elementwise:
      return Elementwise{};
    case PredicateKind::Commutative:
      return Commutative{};
    case PredicateKind::Associative:
      return Associative{};
    case PredicateKind::Involution:
      return Involution{};
    case PredicateKind::LinearIn:
      return LinearIn{*predicate.operand};
    case PredicateKind::HomogeneousIn:
      return HomogeneousIn{*predicate.operand};
    default:
      throw std::logic_error("predicate is not an operator property");
  }
}
PropertyDecision checkNode(const TensorNode& node, TensorFacts result,
                           const std::vector<TensorFacts>& operands,
                           const Predicate& predicate, NumericalPolicy policy,
                           std::optional<NumericalPermissions> permissions) {
  return queryProperty(node, toProperty(predicate),
                       {operands, result, policy, permissions});
}
std::optional<Prepared> prepare(const Term& term, const Bindings& bindings,
                                const TensorEGraph& graph,
                                std::string& rejection) {
  Prepared result;
  if (term.kind == TermKind::TensorVariable) {
    result.reference = graph.find(bindings.tensors.at(term.name));
    result.facts = tensorFacts(graph, *result.reference);
    return result;
  }
  std::vector<TensorFacts> facts;
  for (const auto& child : term.children) {
    auto prepared = prepare(child, bindings, graph, rejection);
    if (!prepared) return std::nullopt;
    facts.push_back(prepared->facts);
    result.children.push_back(std::move(*prepared));
  }
  if (term.kind == TermKind::Scale) {
    const auto scalar_type = facts[0].type;
    const auto target_type = facts[1].type;
    if (!scalar_type || scalar_type.getRank() != 0 || !target_type ||
        scalar_type.getElementType() != target_type.getElementType()) {
      rejection = "RHS scale requires a rank-zero scalar of the tensor dtype";
      return std::nullopt;
    }
    if (target_type.getRank() != 0) {
      Prepared broadcast;
      broadcast.node = {
          OpKind::BroadcastInDim, BroadcastAttrs{{}, target_type}, {0}};
      auto inference = inferTensorNode(
          broadcast.node, std::span<const TensorFacts>(facts.data(), 1));
      if (!inference.valid()) {
        rejection = "RHS scale broadcast: " + inference.reason;
        return std::nullopt;
      }
      broadcast.facts = inference.facts;
      broadcast.children.push_back(std::move(result.children[0]));
      result.children[0] = std::move(broadcast);
      facts[0] = result.children[0].facts;
    }
    result.node = {OpKind::Multiply, NoAttrs{}, {0, 0}};
    auto inference = inferTensorNode(result.node, facts);
    if (!inference.valid()) {
      rejection = "RHS scale: " + inference.reason;
      return std::nullopt;
    }
    result.facts = inference.facts;
    return result;
  }
  if (term.kind == TermKind::ConcreteOperator) {
    auto node = buildConcreteNode(term, bindings, facts, rejection);
    if (!node) return std::nullopt;
    result.node = std::move(*node);
  } else {
    result.node = bindings.operators.at(term.name).front().node;
  }
  // Only arity is relevant to standalone inference; local IDs are placeholders.
  result.node.operands.assign(facts.size(), 0);
  auto inference = inferTensorNode(result.node, facts);
  if (!inference.valid()) {
    rejection = "RHS: " + inference.reason;
    return std::nullopt;
  }
  result.facts = inference.facts;
  if (term.kind == TermKind::OperatorVariable)
    result.operator_variable = term.name;
  return result;
}
// Recheck properties for captured operators at their new operand shapes as
// well as inference. Callback-built trees use exactly this validation path.
InferenceResult validatePrepared(const Prepared& p, const TensorEGraph& graph,
                                 const Rule& rule, NumericalPolicy policy,
                                 const SemanticRuleOptions& options) {
  if (p.reference)
    return {InferenceStatus::Valid, tensorFacts(graph, *p.reference), {}};
  std::vector<TensorFacts> operands;
  for (const auto& child : p.children) {
    auto result = validatePrepared(child, graph, rule, policy, options);
    if (!result.valid()) return result;
    operands.push_back(result.facts);
  }
  auto inferred = inferTensorNode(p.node, operands);
  if (!inferred.valid()) return inferred;
  for (const auto& predicate : rule.predicates) {
    if (!p.operator_variable.empty() && predicate.isProperty() &&
        predicate.subject == p.operator_variable) {
      auto decision = checkNode(p.node, inferred.facts, operands, predicate,
                                policy, options.permissions);
      if (!decision.allowed)
        return {InferenceStatus::Invalid, {}, "RHS: " + decision.reason};
    }
  }
  return inferred;
}
eggc::Id instantiate(const Prepared& prepared, TensorEGraph& graph) {
  if (prepared.reference) return graph.find(*prepared.reference);
  auto node = prepared.node;
  node.operands.clear();
  for (const auto& child : prepared.children)
    node.operands.push_back(instantiate(child, graph));
  return graph.add(std::move(node));
}
std::string_view valueGuardName(PredicateKind kind) {
  switch (kind) {
    case PredicateKind::Scalar:
      return "Scalar";
    case PredicateKind::Uniform:
      return "Uniform";
    case PredicateKind::Rank:
      return "Rank";
    default:
      throw std::logic_error("not a value guard");
  }
}
}  // namespace
TensorRewrite lowerRule(Rule rule, NumericalPolicy policy,
                        SemanticRuleOptions options) {
  using Rewrite = TensorRewrite;
  const auto name = rule.name;
  return Rewrite{
      name, [rule = std::move(rule), policy, options](
                const TensorEGraph& graph, const Rewrite::Sink& sink,
                const eggc::StopCheck& stop) {
        SemanticRuleStats local;
        auto& stats = options.report ? (*options.report)[rule.name] : local;
        SearchState state{stop, options, stats, policy};
        if (state.cancelled()) return false;
        std::unordered_set<std::string> seen;
        const auto* schema = rule.lhs.kind == TermKind::ConcreteOperator
                                 ? opSchema(rule.lhs.op)
                                 : nullptr;
        std::vector<eggc::Id> candidates;
        if (rule.lhs.kind == TermKind::Scale)
          candidates = graph.classes_for_op(OpKind::Multiply);
        else if (schema)
          candidates = graph.classes_for_op(schema->op);
        else {
          // A property on the root operator restricts the discriminant index.
          auto predicate = std::find_if(
              rule.predicates.begin(), rule.predicates.end(),
              [&](const Predicate& p) {
                return p.isProperty() && p.subject == rule.lhs.name;
              });
          if (predicate == rule.predicates.end())
            candidates = graph.classes();
          else {
            std::unordered_set<eggc::Id> roots;
            for (const auto& registered : opSchemas()) {
              auto op = registered.op;
              auto properties = declaredProperties(op);
              if (std::find(properties.begin(), properties.end(),
                            toProperty(*predicate)) == properties.end())
                continue;
              for (auto id : graph.classes_for_op(op)) roots.insert(id);
            }
            candidates.assign(roots.begin(), roots.end());
            std::sort(candidates.begin(), candidates.end());
          }
        }
        for (auto root : candidates) {
          bool complete = matchTerm(
              rule.lhs, root, graph, {}, state, [&](const Bindings& binding) {
                if (state.cancelled()) return false;
                const auto reject = [&](std::string reason) {
                  ++stats.rejections[std::move(reason)];
                  return sink({root, {}, false});
                };
                if (!seen.insert(bindingKey(root, binding)).second) {
                  ++stats.duplicates;
                  return true;
                }
                ++state.matches;
                ++stats.structural_matches;
                for (const auto& predicate : rule.predicates) {
                  if (predicate.isRuleGuard()) continue;
                  if (!predicate.isProperty()) {
                    auto id = binding.tensors.at(predicate.subject);
                    const auto type = tensorFacts(graph, id).type;
                    std::unordered_set<eggc::Id> visited;
                    bool allowed = false;
                    switch (predicate.kind) {
                      case PredicateKind::Scalar:
                        allowed = type && type.getRank() == 0;
                        break;
                      case PredicateKind::Uniform:
                        allowed = isUniformTensor(graph, id, visited, state);
                        break;
                      case PredicateKind::Rank:
                        allowed = type && type.getRank() == *predicate.operand;
                        break;
                      default:
                        throw std::logic_error("not a value predicate");
                    }
                    if ((stop && stop()) || state.budget) return false;
                    if (!allowed)
                      return reject(
                          std::string(valueGuardName(predicate.kind)) +
                          " requires proven value " + predicate.subject);
                    continue;
                  }
                  for (const auto& occurrence :
                       binding.operators.at(predicate.subject)) {
                    std::vector<TensorFacts> operands;
                    for (auto child : occurrence.node.operands)
                      operands.push_back(tensorFacts(graph, child));
                    auto decision = checkNode(
                        occurrence.node, tensorFacts(graph, occurrence.eclass),
                        operands, predicate, policy, options.permissions);
                    if (!decision.allowed) {
                      return reject(decision.reason);
                    }
                  }
                }
                std::string rejection;
                std::optional<Prepared> prepared;
                if (rule.rhs) {
                  prepared = prepare(*rule.rhs, binding, graph, rejection);
                } else {
                  auto match = PatternAccess::match(binding, graph);
                  auto builder = PatternAccess::builder(binding, graph);
                  try {
                    prepared = PatternAccess::prepared(
                        rule.builder(match, builder), rejection);
                  } catch (const std::invalid_argument& error) {
                    rejection = error.what();
                  }
                }
                if (!prepared ||
                    prepared->facts.type != tensorFacts(graph, root).type) {
                  return reject(
                      rejection.empty()
                          ? "RHS result type differs from matched root"
                          : rejection);
                }
                auto checked =
                    validatePrepared(*prepared, graph, rule, policy, options);
                if (!checked.valid()) return reject(checked.reason);
                for (const auto& predicate : rule.predicates) {
                  if (!predicate.isRuleGuard()) continue;
                  auto decision = checkRuleGuard(predicate, binding, *prepared,
                                                 graph, policy, options);
                  if (!decision.allowed) return reject(decision.reason);
                }
                if (stop && stop()) return false;
                ++stats.accepted;
                return sink(
                    {root,
                     [prepared = std::move(*prepared), report = options.report,
                      name = rule.name, root, binding, rule, policy,
                      options](TensorEGraph& mutable_graph)
                         -> std::optional<eggc::Id> {
                       // Recheck the entire prepared tree against current facts
                       // before any add.
                       auto inferred = validatePrepared(prepared, mutable_graph,
                                                        rule, policy, options);
                       if (!inferred.valid() ||
                           inferred.facts.type !=
                               tensorFacts(mutable_graph, root).type)
                         return std::nullopt;
                       for (const auto& predicate : rule.predicates)
                         if (predicate.isRuleGuard() &&
                             !checkRuleGuard(predicate, binding, prepared,
                                             mutable_graph, policy, options)
                                  .allowed)
                           return std::nullopt;
                       if (report) ++(*report)[name].applied;
                       return instantiate(prepared, mutable_graph);
                     },
                     true});
              });
          if (!complete) return false;
        }
        return true;
      }};
}
}  // namespace semantic_detail

}  // namespace joint_shard

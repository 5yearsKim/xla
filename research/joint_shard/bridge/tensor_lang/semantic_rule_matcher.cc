#include <algorithm>

#include "llvm/Support/Casting.h"
#include "research/joint_shard/bridge/tensor_lang/semantic_rule_internal.h"

namespace joint_shard {
namespace semantic_detail {
// Follow only broadcasts: this keeps scalar identity explicit and does not
// invent a scalar from a tensor splat. All helper searches share rule budgets.
using ScalarSink = std::function<bool(eggc::Id)>;
bool findScalars(eggc::Id id, const TensorEGraph& graph, SearchState& state,
                 std::unordered_set<eggc::Id>& seen, const ScalarSink& sink) {
  if (state.cancelled()) return false;
  id = graph.find(id);
  if (!seen.insert(id).second) return true;
  const auto& facts = tensorFacts(graph, id);
  if (facts.type && facts.type.getRank() == 0) return sink(id);
  for (const auto& node : graph.nodes(id)) {
    if (state.cancelled()) return false;
    ++state.visits;
    ++state.stats.node_visits;
    if (node.op == OpKind::BroadcastInDim &&
        !findScalars(node.operands[0], graph, state, seen, sink))
      return false;
  }
  return true;
}
// Depth-first streaming matching; no intermediate Cartesian-product lists.
bool matchTerm(const Term& term, eggc::Id id, const TensorEGraph& graph,
               const Bindings& initial, SearchState& state,
               const BindingSink& sink) {
  if (state.cancelled()) return false;
  id = graph.find(id);
  if (term.variable) {
    Bindings next = initial;
    auto [found, inserted] = next.tensors.emplace(term.name, id);
    if (!inserted && graph.find(found->second) != id) return true;
    return sink(next);
  }
  if (term.name == "scale") {
    for (const auto& node : graph.nodes(id)) {
      if (state.cancelled()) return false;
      ++state.visits;
      ++state.stats.node_visits;
      if (node.op != OpKind::Multiply) continue;
      // The RHS builder puts the scalar first. Matching a scalar on the right
      // therefore also needs permission to commute this multiply (e.g. strict
      // floating evaluation cannot freely reorder NaN operands).
      for (unsigned slot = 0; slot < 2; ++slot) {
        if (slot == 1) {
          std::vector<TensorFacts> operands{
              tensorFacts(graph, node.operands[0]),
              tensorFacts(graph, node.operands[1])};
          if (!queryProperty(node, Commutative{},
                             {operands, tensorFacts(graph, id), state.policy,
                              state.options.permissions})
                   .allowed)
            continue;
        }
        std::unordered_set<eggc::Id> seen;
        if (!findScalars(
                node.operands[slot], graph, state, seen, [&](eggc::Id scalar) {
                  return matchTerm(term.children[0], scalar, graph, initial,
                                   state, [&](const Bindings& bound) {
                                     return matchTerm(term.children[1],
                                                      node.operands[1 - slot],
                                                      graph, bound, state,
                                                      sink);
                                   });
                }))
          return false;
      }
    }
    return true;
  }
  const auto* schema = lookupOpSchema(term.name);
  for (const auto& node : graph.nodes(id)) {
    if (state.cancelled()) return false;
    ++state.visits;
    ++state.stats.node_visits;
    if (node.operands.size() != term.children.size()) continue;
    if (schema &&
        (node.op != schema->op || !std::holds_alternative<NoAttrs>(node.attrs)))
      continue;
    Bindings bound = initial;
    if (!schema) {
      auto& occurrences = bound.operators[term.name];
      if (!occurrences.empty() && !occurrences.front().node.matches(node))
        continue;
      occurrences.push_back({node, id});
    }
    const auto children = [&](auto&& self, size_t index,
                              const Bindings& binding) -> bool {
      if (state.cancelled()) return false;
      if (index == term.children.size()) return sink(binding);
      return matchTerm(
          term.children[index], node.operands[index], graph, binding, state,
          [&](const Bindings& next) { return self(self, index + 1, next); });
    };
    if (!children(children, 0, bound)) return false;
  }
  return true;
}
bool isUniformTensor(const TensorEGraph& graph, eggc::Id id,
                     std::unordered_set<eggc::Id>& seen, SearchState& state) {
  if (state.cancelled()) return false;
  id = graph.find(id);
  if (!seen.insert(id).second) return false;
  const auto& facts = tensorFacts(graph, id);
  if (facts.type && facts.type.getRank() == 0) return true;
  if (auto value =
          llvm::dyn_cast_or_null<mlir::DenseElementsAttr>(facts.constant))
    if (value.isSplat()) return true;
  for (const auto& node : graph.nodes(id)) {
    if (state.cancelled()) return false;
    ++state.visits;
    ++state.stats.node_visits;
    if (node.op == OpKind::BroadcastInDim &&
        isUniformTensor(graph, node.operands[0], seen, state))
      return true;
  }
  return false;
}
std::string bindingKey(eggc::Id root, const Bindings& bindings) {
  std::vector<std::string> entries;
  for (const auto& [name, id] : bindings.tensors)
    entries.push_back(name + "=" + std::to_string(id));
  for (const auto& [name, occurrences] : bindings.operators)
    for (const auto& occurrence : occurrences)
      entries.push_back(name + "=" + std::to_string(occurrence.eclass) + ":" +
                        occurrence.node.format());
  std::sort(entries.begin(), entries.end());
  std::string key = std::to_string(root);
  for (const auto& entry : entries)
    key += "|" + std::to_string(entry.size()) + ":" + entry;
  return key;
}
}  // namespace semantic_detail

}  // namespace joint_shard

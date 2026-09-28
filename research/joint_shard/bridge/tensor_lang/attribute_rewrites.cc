#include <algorithm>

#include "research/joint_shard/bridge/tensor_lang/tensor_rewrites.h"

namespace joint_shard {

namespace {
bool identity(const std::vector<int64_t>& p) {
  for (size_t i = 0; i < p.size(); ++i)
    if (p[i] != static_cast<int64_t>(i)) return false;
  return true;
}
bool emitNode(const TensorEGraph& graph, eggc::Id root, TensorNode node,
              const TensorRewrite::Sink& sink) {
  std::vector<TensorFacts> facts;
  for (auto child : node.operands) facts.push_back(tensorFacts(graph, child));
  auto inference = inferTensorNode(node, facts);
  if (!inference.valid() ||
      inference.facts.type != tensorFacts(graph, root).type)
    return true;
  return sink({root,
               [node = std::move(node)](
                   TensorEGraph& target) -> std::optional<eggc::Id> {
                 std::vector<TensorFacts> facts;
                 for (auto child : node.operands)
                   facts.push_back(tensorFacts(target, child));
                 if (!inferTensorNode(node, facts).valid()) return std::nullopt;
                 return target.add(node);
               }});
}
}  // namespace
std::vector<TensorRewrite> buildAttributeRewrites() {
  return {
      TensorRewrite{
          "compose-shape-operators",
          [](const TensorEGraph& graph, const TensorRewrite::Sink& sink,
             const eggc::StopCheck& stop) {
            size_t visits = 0;
            const auto cancelled = [&] {
              return (stop && stop()) || ++visits > 100000;
            };
            for (auto op :
                 {OpKind::Transpose, OpKind::Reshape, OpKind::BroadcastInDim})
              for (auto root : graph.classes_for_op(op))
                for (const auto& outer : graph.nodes(root)) {
                  if (cancelled()) return false;
                  if (outer.op != op) continue;
                  auto child = outer.operands[0];
                  bool noop = false;
                  if (auto* p = std::get_if<TransposeAttrs>(&outer.attrs))
                    noop = identity(p->permutation);
                  if (std::holds_alternative<ReshapeAttrs>(outer.attrs))
                    noop = true;
                  if (auto* b = std::get_if<BroadcastAttrs>(&outer.attrs))
                    noop = identity(b->dimensions);
                  if (noop && tensorFacts(graph, child).type ==
                                  tensorFacts(graph, root).type)
                    if (!sink({root,
                               [child](TensorEGraph& graph)
                                   -> std::optional<eggc::Id> {
                                 return graph.find(child);
                               }}))
                      return false;
                  for (const auto& inner : graph.nodes(child)) {
                    if (cancelled()) return false;
                    if (inner.op != op) continue;
                    auto candidate = outer;
                    candidate.operands = inner.operands;
                    if (op == OpKind::Transpose) {
                      auto& p =
                          std::get<TransposeAttrs>(candidate.attrs).permutation;
                      const auto& q =
                          std::get<TransposeAttrs>(inner.attrs).permutation;
                      if (p.size() != q.size()) continue;
                      for (auto& axis : p) axis = q[axis];
                    }
                    if (op == OpKind::BroadcastInDim) {
                      auto& p =
                          std::get<BroadcastAttrs>(candidate.attrs).dimensions;
                      const auto& q =
                          std::get<BroadcastAttrs>(inner.attrs).dimensions;
                      std::vector<int64_t> combined;
                      for (auto axis : q) combined.push_back(p[axis]);
                      p = std::move(combined);
                    }
                    if (!emitNode(graph, root, std::move(candidate), sink))
                      return false;
                  }
                }
            return true;
          }},
      TensorRewrite{
          "absorb-dot-input-transpose",
          [](const TensorEGraph& graph, const TensorRewrite::Sink& sink,
             const eggc::StopCheck& stop) {
            size_t visits = 0;
            for (auto root : graph.classes_for_op(OpKind::DotGeneral))
              for (const auto& dot : graph.nodes(root)) {
                if (dot.op != OpKind::DotGeneral) continue;
                for (unsigned side = 0; side < 2; ++side)
                  for (const auto& transpose :
                       graph.nodes(dot.operands[side])) {
                    if ((stop && stop()) || ++visits > 100000) return false;
                    if (transpose.op != OpKind::Transpose) continue;
                    auto candidate = dot;
                    auto& attrs = std::get<DotGeneralAttrs>(candidate.attrs);
                    auto& contract = side == 0 ? attrs.lhs_contracting
                                               : attrs.rhs_contracting;
                    auto& batch =
                        side == 0 ? attrs.lhs_batching : attrs.rhs_batching;
                    const auto& permutation =
                        std::get<TransposeAttrs>(transpose.attrs).permutation;
                    // Free axes must retain output order. Otherwise an output
                    // transpose would be required; leave that richer
                    // transformation to a new rule.
                    std::vector<int64_t> free;
                    for (int64_t axis = 0;
                         axis < static_cast<int64_t>(permutation.size());
                         ++axis)
                      if (std::find(contract.begin(), contract.end(), axis) ==
                              contract.end() &&
                          std::find(batch.begin(), batch.end(), axis) ==
                              batch.end())
                        free.push_back(permutation[axis]);
                    if (!std::is_sorted(free.begin(), free.end())) continue;
                    for (auto& axis : contract) axis = permutation[axis];
                    for (auto& axis : batch) axis = permutation[axis];
                    candidate.operands[side] = transpose.operands[0];
                    if (!emitNode(graph, root, std::move(candidate), sink))
                      return false;
                  }
              }
            return true;
          }}};
}

}  // namespace joint_shard

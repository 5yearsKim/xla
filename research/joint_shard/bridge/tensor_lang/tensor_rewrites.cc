#include "research/joint_shard/bridge/tensor_lang/tensor_rewrites.h"

#include <array>
#include <utility>
TensorCondition sameTensorType(std::string lhs, std::string rhs) {
  return {"same-tensor-type",
          {lhs, rhs},
          [lhs = std::move(lhs), rhs = std::move(rhs)](
              const TensorEGraph& graph, eggc::Id,
              const TensorSubstitution& subst) {
            auto a = subst.find(lhs), b = subst.find(rhs);
            if (a == subst.end() || b == subst.end()) return false;
            auto at = tensorFacts(graph, a->second).type,
                 bt = tensorFacts(graph, b->second).type;
            return at && bt && at.hasStaticShape() && bt.hasStaticShape() &&
                   at == bt;
          }};
}
std::vector<TensorRewrite> buildTensorRewrites(NumericalPolicy policy) {
  auto x = TensorPattern::var("x"), y = TensorPattern::var("y");
  auto sameType = sameTensorType("?x", "?y");
  TensorCondition guard{
      "commutative-same-type",
      {"?x", "?y"},
      [sameType, policy](const TensorEGraph& graph, eggc::Id root,
                         const TensorSubstitution& subst) {
        if (!sameType.check(graph, root, subst)) return false;
        TensorNode node{
            OpKind::Add, NoAttrs{}, {subst.at("?x"), subst.at("?y")}};
        const std::array operands = {tensorFacts(graph, node.operands[0]),
                                     tensorFacts(graph, node.operands[1])};
        return hasProperty(
            node, Commutative{},
            PropertyContext{operands, tensorFacts(graph, root), policy});
      }};
  // Multiply commutativity remains disabled until explicitly requested.
  return {
      {"commute-add",
       TensorPattern::node(TensorNode{OpKind::Add, NoAttrs{}, {0, 0}}, {x, y}),
       TensorPattern::node(TensorNode{OpKind::Add, NoAttrs{}, {0, 0}}, {y, x}),
       guard}};
}

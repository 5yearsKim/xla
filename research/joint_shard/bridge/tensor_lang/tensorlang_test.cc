#include "research/joint_shard/bridge/tensor_lang/tensorlang.h"

#include <stdexcept>
#include <vector>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "eggc/egraph.hpp"
#include "eggc/extract.hpp"
#include "eggc/runner.hpp"
#include "gtest/gtest.h"
#include "research/joint_shard/bridge/tensor_lang/tensor_analysis.h"
#include "research/joint_shard/bridge/tensor_lang/tensor_rewrites.h"

namespace joint_shard {

static_assert(eggc::Language<TensorNode>);
static_assert(eggc::AnalysisFor<TensorAnalysis, TensorNode>);

namespace {

class TensorLangTest : public ::testing::Test {
 protected:
  mlir::RankedTensorType tensor(std::vector<int64_t> shape) {
    return mlir::RankedTensorType::get(shape,
                                       mlir::Float32Type::get(&context_));
  }

  mlir::MLIRContext context_;
};

TEST_F(TensorLangTest, IdentityIncludesNativeAttributesAndResultType) {
  eggc::EGraph<TensorNode> graph;
  auto x =
      graph.add(TensorNode{OpKind::Input, InputAttrs{0, tensor({2, 3})}, {}});
  auto a =
      graph.add(TensorNode{OpKind::Reshape, ReshapeAttrs{tensor({6})}, {x}});
  auto b =
      graph.add(TensorNode{OpKind::Reshape, ReshapeAttrs{tensor({3, 2})}, {x}});
  EXPECT_NE(a, b);
  EXPECT_EQ(a, graph.add(TensorNode{
                   OpKind::Reshape, ReshapeAttrs{tensor({6})}, {x}}));
  auto t =
      graph.add(TensorNode{OpKind::Transpose, TransposeAttrs{{0, 1}}, {x}});
  auto u =
      graph.add(TensorNode{OpKind::Transpose, TransposeAttrs{{1, 0}}, {x}});
  EXPECT_NE(t, u);
  graph.rebuild();
  EXPECT_NE(graph.find(t), graph.find(u));
  auto pattern = TensorPattern::node(
      TensorNode{OpKind::Transpose, TransposeAttrs{{0, 1}}, {0}},
      {TensorPattern::var("x")});
  EXPECT_EQ(eggc::match(graph, pattern, t).size(), 1);
  EXPECT_TRUE(eggc::match(graph, pattern, u).empty());
}

TEST_F(TensorLangTest, InlineInputsRejectTypeConflicts) {
  TensorEGraph graph(TensorAnalysis{});
  auto a =
      graph.add(TensorNode{OpKind::Input, InputAttrs{0, tensor({2, 3})}, {}});
  auto b =
      graph.add(TensorNode{OpKind::Input, InputAttrs{1, tensor({3, 2})}, {}});
  auto integer = mlir::RankedTensorType::get(
      {2, 3}, mlir::IntegerType::get(&context_, 32));
  auto c = graph.add(TensorNode{OpKind::Input, InputAttrs{2, integer}, {}});
  EXPECT_EQ(tensorFacts(graph, a).type, tensor({2, 3}));
  EXPECT_THROW(graph.merge(a, b), eggc::AnalysisConflict);
  EXPECT_THROW(graph.merge(a, c), eggc::AnalysisConflict);
  EXPECT_NE(graph.find(a), graph.find(b));
  EXPECT_THROW((graph.add(TensorNode{OpKind::Input, InputAttrs{3, {}}, {}})),
               std::invalid_argument);
}

TEST_F(TensorLangTest, RejectsBadOperandsBeforeNodeInsertion) {
  TensorEGraph graph(TensorAnalysis{});
  auto type = tensor({2, 3});
  auto a = graph.add(TensorNode{OpKind::Input, InputAttrs{0, type}, {}});
  auto b =
      graph.add(TensorNode{OpKind::Input, InputAttrs{1, tensor({3, 2})}, {}});
  auto add = OpKind::Add;
  auto before = graph.node_count();
  EXPECT_THROW((graph.add(TensorNode{add, NoAttrs{}, {a, b}})),
               eggc::AnalysisConflict);
  EXPECT_THROW((graph.add(TensorNode{add, NoAttrs{}, {a}})),
               std::invalid_argument);
  EXPECT_EQ(graph.node_count(), before);
}

TEST_F(TensorLangTest, AnalysisMergeRefinesUnknownFacts) {
  TensorAnalysis analysis;
  TensorFacts unknown = TensorFacts{};
  TensorFacts known = TensorFacts{tensor({2, 3})};
  EXPECT_EQ(analysis.merge(unknown, known), eggc::AnalysisMerge::Changed);
  EXPECT_EQ(analysis.merge(unknown, known), eggc::AnalysisMerge::Unchanged);
  EXPECT_EQ(unknown.type, tensor({2, 3}));
}

TEST_F(TensorLangTest, SemanticRulesCommuteStaticAdd) {
  TensorEGraph graph(TensorAnalysis{});
  auto type = tensor({2, 3});
  auto a = graph.add(TensorNode{OpKind::Input, InputAttrs{0, type}, {}});
  auto b = graph.add(TensorNode{OpKind::Input, InputAttrs{1, type}, {}});
  auto add = OpKind::Add;
  auto root = graph.add(TensorNode{add, NoAttrs{}, {a, b}});
  auto rules = parseSemanticRules(defaultSemanticRules(),
                                  NumericalPolicy::AllowReassociation);
  auto report = eggc::run(graph, rules);
  EXPECT_EQ(report.reason, eggc::StopReason::Saturated);
  auto reversed = graph.add(TensorNode{add, NoAttrs{}, {b, a}});
  EXPECT_EQ(graph.find(root), graph.find(reversed));
  graph.rebuild();
}

TEST_F(TensorLangTest, DynamicShapeSyntaxDoesNotProveRuntimeShapeEquality) {
  TensorEGraph graph(TensorAnalysis{});
  auto type = tensor({mlir::ShapedType::kDynamic, 3});
  auto a = graph.add(TensorNode{OpKind::Input, InputAttrs{0, type}, {}});
  auto b = graph.add(TensorNode{OpKind::Input, InputAttrs{1, type}, {}});
  auto add = OpKind::Add;
  auto root = graph.add(TensorNode{add, NoAttrs{}, {a, b}});
  auto report = eggc::run(graph, parseSemanticRules(defaultSemanticRules()));
  ASSERT_FALSE(report.history.empty());
  EXPECT_GT(report.history[0].condition_rejections, 0);
  auto reversed = graph.add(TensorNode{add, NoAttrs{}, {b, a}});
  EXPECT_NE(graph.find(root), graph.find(reversed));
}

TEST_F(TensorLangTest, CustomSearchRetainsEachMatchedAttributeAlternative) {
  using Graph = eggc::EGraph<TensorNode>;
  using Rule = eggc::Rewrite<TensorNode>;
  Graph graph;
  auto x =
      graph.add(TensorNode{OpKind::Input, InputAttrs{0, tensor({2, 2})}, {}});
  auto y =
      graph.add(TensorNode{OpKind::Input, InputAttrs{1, tensor({2, 2})}, {}});
  auto a =
      graph.add(TensorNode{OpKind::Transpose, TransposeAttrs{{0, 1}}, {x}});
  auto b =
      graph.add(TensorNode{OpKind::Transpose, TransposeAttrs{{1, 0}}, {x}});
  graph.merge(a, b);
  graph.rebuild();
  // Synthetic equivalences exercise the extension interface, not a tensor rule.
  std::vector<OpAttrs> matchedAttrs;
  Rule rule("copy-transpose-alternatives", [&, y](const Graph& snapshot,
                                                  const Rule::Sink& emit,
                                                  const eggc::StopCheck& stop) {
    for (auto root : snapshot.classes_for_op(OpKind::Transpose)) {
      for (auto node : snapshot.nodes(root)) {
        if (stop && stop()) return false;
        matchedAttrs.push_back(node.attrs);
        node.operands = {y};
        if (!emit({root, [node](Graph& target) -> std::optional<eggc::Id> {
                     return target.add(node);
                   }}))
          return false;
      }
    }
    return true;
  });
  eggc::run(graph, std::vector<Rule>{rule}, 1);
  ASSERT_EQ(matchedAttrs.size(), 2);
  EXPECT_FALSE(matchedAttrs[0] == matchedAttrs[1]);
  EXPECT_EQ(graph.nodes(a).size(), 4);
  auto first =
      graph.add(TensorNode{OpKind::Transpose, TransposeAttrs{{0, 1}}, {y}});
  auto second =
      graph.add(TensorNode{OpKind::Transpose, TransposeAttrs{{1, 0}}, {y}});
  EXPECT_EQ(graph.find(first), graph.find(a));
  EXPECT_EQ(graph.find(second), graph.find(a));
}

}  // namespace

}  // namespace joint_shard

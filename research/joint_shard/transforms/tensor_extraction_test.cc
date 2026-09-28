#include "research/joint_shard/transforms/tensor_extraction.h"

#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

namespace joint_shard {

namespace {
class TensorExtractionTest : public ::testing::Test {
 protected:
  mlir::MLIRContext context;
  TensorEGraph graph;
  eggc::DagOptions budgets;

  mlir::RankedTensorType tensor(std::vector<int64_t> shape) {
    return mlir::RankedTensorType::get(shape,
                                       mlir::IntegerType::get(&context, 32));
  }
  eggc::Id input(unsigned index, std::vector<int64_t> shape) {
    return graph.add({OpKind::Input, InputAttrs{index, tensor(shape)}, {}});
  }
  TensorNode dot(eggc::Id a, eggc::Id b) {
    return {OpKind::DotGeneral,
            DotGeneralAttrs{
                DotDimensions{{1}, {0}, {}, {}}, {}, {}, tensor({2, 1}), {}},
            {a, b}};
  }
  // Modular i32 distributivity: dot(a,b) + dot(a,b) == dot(a+a,b).
  // Tree cost favors factoring, but DAG cost favors reusing the small output
  // instead of adding the larger input. Both forms are valid tensor equations.
  eggc::Id alternatives(bool expensive_first = false) {
    // Make e-class IDs differ from expression indices, exposing accidental
    // graph lookups while scoring a RecExpr.
    input(99, {7});
    auto a = input(0, {2, 4});
    auto b = input(1, {4, 1});
    auto doubled = graph.add({OpKind::Add, NoAttrs{}, {a, a}});
    auto negativeA = graph.add({OpKind::Negate, NoAttrs{}, {doubled}});
    auto negativeB = graph.add({OpKind::Negate, NoAttrs{}, {b}});
    auto expensive = graph.add(dot(negativeA, negativeB));
    auto factored = graph.add(dot(doubled, b));
    auto product = graph.add(dot(a, b));
    auto shared = graph.add({OpKind::Add, NoAttrs{}, {product, product}});
    // Preserve insertion order deliberately: the first finite DAG can be
    // worse than the baseline when a budget interrupts search.
    if (expensive_first) {
      graph.merge(expensive, factored);
      graph.merge(expensive, shared);
    } else {
      graph.merge(shared, factored);
    }
    graph.rebuild();
    return shared;
  }
  TensorRecExpr extract(
      eggc::Id root, TensorExtractionReport& report,
      TensorExtractorMode mode = TensorExtractorMode::Auto,
      ExtractionProfile profile = ExtractionProfile::Compute) {
    return extractTensorRoots(graph, {root}, profile, mode, budgets, report)
        .front();
  }
};

TEST_F(TensorExtractionTest, SelectsCheaperSharedComputation) {
  auto root = alternatives();
  TensorExtractionReport treeReport, dagReport;
  auto tree = extract(root, treeReport, TensorExtractorMode::Tree);
  auto dag = extract(root, dagReport);
  EXPECT_EQ(tree.nodes.back().op, OpKind::DotGeneral);
  ASSERT_EQ(dag.nodes.back().op, OpKind::Add);
  EXPECT_EQ(dag.nodes.back().operands[0], dag.nodes.back().operands[1]);
  EXPECT_TRUE(dagReport.selected_dag);
  EXPECT_TRUE(dagReport.optimal);
  EXPECT_EQ(dagReport.baseline_cost, 26.0);
  EXPECT_EQ(dagReport.candidate_cost, 20.0);
}

TEST_F(TensorExtractionTest, StateBudgetWithoutCandidateKeepsBaseline) {
  auto root = alternatives();
  budgets.state_limit = 1;
  TensorExtractionReport report;
  auto expression = extract(root, report);
  EXPECT_EQ(expression.nodes.back().op, OpKind::DotGeneral);
  EXPECT_TRUE(report.attempted_dag);
  EXPECT_FALSE(report.selected_dag);
  EXPECT_FALSE(report.candidate_cost);
  EXPECT_EQ(report.stop, eggc::DagStopReason::StateLimit);
  EXPECT_EQ(report.fallback, "no-dag-candidate");
}

TEST_F(TensorExtractionTest, FrontierAndTimeBudgetsKeepBaseline) {
  auto root = alternatives();
  budgets.frontier_limit = 1;
  TensorExtractionReport report;
  extract(root, report);
  EXPECT_EQ(report.stop, eggc::DagStopReason::FrontierLimit);
  EXPECT_FALSE(report.selected_dag);
  budgets.frontier_limit = 1000;
  // A zero time budget deterministically cancels before expanding a state.
  budgets.time_limit = std::chrono::milliseconds(0);
  extract(root, report);
  EXPECT_EQ(report.stop, eggc::DagStopReason::TimeLimit);
  EXPECT_EQ(report.explored_states, 0);
  EXPECT_FALSE(report.selected_dag);
}

TEST_F(TensorExtractionTest, RejectsWorsePartialCandidate) {
  auto root = alternatives(true);
  bool observed = false;
  for (size_t limit = 1; limit < 100; ++limit) {
    budgets.state_limit = limit;
    TensorExtractionReport report;
    auto expression = extract(root, report);
    if (!report.candidate_cost ||
        *report.candidate_cost <= *report.baseline_cost)
      continue;
    EXPECT_FALSE(report.optimal);
    EXPECT_FALSE(report.selected_dag);
    EXPECT_EQ(report.fallback, "baseline-no-worse");
    EXPECT_EQ(expression.nodes.back().op, OpKind::DotGeneral);
    observed = true;
    break;
  }
  EXPECT_TRUE(observed);
}

TEST_F(TensorExtractionTest, AcceptsImprovedPartialCandidate) {
  auto root = alternatives();
  // Leave another equivalent alternative after the best candidate, so search
  // can stop with that incumbent before exhausting all alternatives.
  auto a = input(0, {2, 4});
  auto b = input(1, {4, 1});
  auto negativeA = graph.add({OpKind::Negate, NoAttrs{}, {a}});
  auto doubledB = graph.add({OpKind::Add, NoAttrs{}, {b, b}});
  auto negativeB = graph.add({OpKind::Negate, NoAttrs{}, {doubledB}});
  graph.merge(root, graph.add(dot(negativeA, negativeB)));
  graph.rebuild();
  bool observed = false;
  for (size_t limit = 1; limit < 100; ++limit) {
    budgets.state_limit = limit;
    TensorExtractionReport report;
    extract(root, report);
    if (report.selected_dag && !report.optimal) {
      EXPECT_LT(*report.candidate_cost, *report.baseline_cost);
      EXPECT_EQ(report.stop, eggc::DagStopReason::StateLimit);
      observed = true;
      break;
    }
  }
  EXPECT_TRUE(observed);
}

TEST_F(TensorExtractionTest, TiesKeepBaseline) {
  auto x = input(0, {2, 1});
  auto root = graph.add({OpKind::Add, NoAttrs{}, {x, x}});
  graph.rebuild();
  TensorExtractionReport report;
  extract(root, report);
  EXPECT_TRUE(report.optimal);
  EXPECT_EQ(report.candidate_cost, report.baseline_cost);
  EXPECT_FALSE(report.selected_dag);
  EXPECT_EQ(report.fallback, "baseline-no-worse");
}

TEST_F(TensorExtractionTest, OtherProfilesAndMultipleOutputsUseTree) {
  auto root = alternatives();
  for (auto profile : {ExtractionProfile::Depth, ExtractionProfile::Memory}) {
    TensorExtractionReport automatic, tree;
    auto a = extract(root, automatic, TensorExtractorMode::Auto, profile);
    auto b = extract(root, tree, TensorExtractorMode::Tree, profile);
    EXPECT_EQ(a.nodes, b.nodes);
    EXPECT_FALSE(automatic.attempted_dag);
    EXPECT_EQ(automatic.fallback, "non-compute-profile");
  }
  TensorExtractionReport report;
  auto outputs =
      extractTensorRoots(graph, {root, root}, ExtractionProfile::Compute,
                         TensorExtractorMode::Auto, budgets, report);
  EXPECT_EQ(outputs.size(), 2);
  EXPECT_FALSE(report.attempted_dag);
  EXPECT_EQ(report.fallback, "multiple-outputs");
}
}  // namespace

}  // namespace joint_shard

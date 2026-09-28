#include "research/joint_shard/search/region_summary.h"

#include "research/joint_shard/tests/support/region_test.h"

namespace joint_shard {
namespace {
using RegionSummaryTest = test::RegionTest;
TEST_F(RegionSummaryTest, ExactBoundarySelectionPreservesOriginalOnTies) {
  auto module = scalingDot();
  auto mesh = selectMesh(*module);
  auto type =
      mlir::RankedTensorType::get({8, 4}, mlir::Float32Type::get(&context));
  auto choices = tensorLayoutChoices(type, mesh, {}, {});
  BoundaryState first{{choices[0]}, {choices[0]}},
      second{{choices[1]}, {choices[1]}};
  RegionSummary summary;
  summary.record(0, first, {true, {12, 0, 0}, {}, "original"});
  summary.record(1, first, {true, {8, 0, 0}, {}, "winner"});
  summary.record(2, first, {true, {10, 0, 0}, {}, "loser"});
  summary.record(0, second, {true, {5, 0, 0}, {}, "original"});
  summary.record(3, second, {true, {5, 0, 0}, {}, "tie"});
  ASSERT_EQ(summary.plans.size(), 2);
  EXPECT_EQ(summary.evaluations, 5);
  EXPECT_EQ(summary.feasible_plans, 5);
  summary.finalizePlans();
  ASSERT_EQ(summary.plans.size(), 2);
  for (const auto& plan : summary.plans)
    EXPECT_EQ(plan.candidate_id, plan.boundary == first ? 1 : 0);
}

TEST_F(RegionSummaryTest, UnknownEstimatesDoNotDisplaceKnownPlans) {
  auto module = scalingDot();
  auto state =
      enumerateInputStates(regions(*module)[0], selectMesh(*module)).states[0];
  BoundaryState boundary{state, {}};
  RegionSummary summary;
  summary.record(0, boundary, {true, {0, 0, 1}, {}, "unknown"});
  summary.record(1, boundary, {true, {5, 0, 0}, {}, "known"});
  summary.record(2, boundary, {true, {0, 0, 1}, {}, "unknown"});
  ASSERT_EQ(summary.plans.size(), 1);
  EXPECT_EQ(summary.plans[0].lowered_mlir, "known");
  summary.finalizePlans();
  ASSERT_EQ(summary.plans.size(), 1);
  EXPECT_EQ(summary.plans[0].candidate_id, 1);
  EXPECT_TRUE(summary.plans[0].cost.known());
}

TEST_F(RegionSummaryTest,
       DominanceUsesInputAndOutputDirectionsAndRetainedWitness) {
  auto module = scalingDot();
  auto mesh = selectMesh(*module);
  auto type =
      mlir::RankedTensorType::get({8, 4}, mlir::Float32Type::get(&context));
  auto choices = tensorLayoutChoices(type, mesh, {}, {});
  BoundaryState a{{choices[0]}, {choices[0]}}, b{{choices[1]}, {choices[1]}};
  RegionSummary summary;
  summary.interface.inputs = {{0, type}};
  summary.interface.outputs = {{1, type}};
  summary.plans = {{0, a, {10, 0, 0}, "expensive"},
                   {1, b, {6, 0, 0}, "survivor"}};
  std::vector<std::pair<TensorSharding, TensorSharding>> requests;
  summary.finalizePlans();
  pruneDominatedStates(summary, [&](auto from, auto to, auto tensor) {
    EXPECT_EQ(tensor, type);
    requests.emplace_back(from, to);
    return Cost{0, from == to ? 0.0 : 1.0, 0};
  });
  ASSERT_EQ(summary.frontier.size(), 1);
  EXPECT_EQ(summary.plans.at(summary.frontier[0]).boundary, b);
  EXPECT_EQ(summary.plans.size(), 2);
  ASSERT_EQ(summary.dominance.size(), 1);
  EXPECT_EQ(summary.dominance[0].replacement, b);
  EXPECT_EQ(summary.dominance[0].replacement_total, 8);
  auto witness_id = summary.dominance[0].replacement_id;
  EXPECT_EQ(summary.plans.at(witness_id).boundary, b);
  EXPECT_TRUE(
      summary.plans[summary.dominance[0].removed_id].lowered_mlir.empty());
  EXPECT_EQ(summary.plans[witness_id].lowered_mlir, "survivor");
  ASSERT_EQ(requests.size(), 2);
  EXPECT_EQ(requests[0], std::make_pair(choices[0], choices[1]));
  EXPECT_EQ(requests[1], std::make_pair(choices[1], choices[0]));
}

TEST_F(RegionSummaryTest,
       DifferentInferredOutputsRemainSeparateUntilDominance) {
  auto module = scalingDot();
  auto mesh = selectMesh(*module);
  auto type =
      mlir::RankedTensorType::get({8, 4}, mlir::Float32Type::get(&context));
  auto choices = tensorLayoutChoices(type, mesh, {}, {});
  RegionSummary summary;
  summary.interface = {{{0, type}}, {{1, type}}};
  summary.record(0, {{choices[0]}, {choices[0]}},
                 {true, {8, 0, 0}, {}, "replicated"});
  summary.record(1, {{choices[0]}, {choices[2]}},
                 {true, {10, 0, 0}, {}, "sharded"});
  summary.finalizePlans();
  ASSERT_EQ(summary.plans.size(), 2);
  pruneDominatedStates(summary, [](auto from, auto to, auto) {
    return Cost{0, from == to ? 0.0 : 5.0, 0};
  });
  EXPECT_EQ(summary.frontier.size(), 2);
  EXPECT_TRUE(summary.dominance.empty());
}

TEST_F(RegionSummaryTest, DominanceChargesEveryOutputAtALiveCut) {
  auto module = scalingDot();
  auto mesh = selectMesh(*module);
  auto type =
      mlir::RankedTensorType::get({8, 4}, mlir::Float32Type::get(&context));
  auto layouts = tensorLayoutChoices(type, mesh, {}, {});
  auto r = layouts[0], tp = layouts[2];
  RegionInterface cut{{}, {{1, type}, {2, type}}};
  std::vector<BoundaryState> boundaries{{{}, {r, r}}, {{}, {tp, tp}}};
  std::vector<Cost> costs{{1, 0, 0}, {8, 0, 0}};
  auto estimate = [](auto from, auto to, auto) {
    return Cost{0, from == to ? 0.0 : 4.0, 0};
  };
  auto result = pruneBoundaryPlans(cut, boundaries, costs, estimate);
  EXPECT_EQ(result.frontier.size(), 2);
  EXPECT_TRUE(result.dominance.empty());
  costs[1] = {10, 0, 0};
  result = pruneBoundaryPlans(cut, boundaries, costs, estimate);
  ASSERT_EQ(result.dominance.size(), 1);
  EXPECT_EQ(result.dominance[0].adapters.total(), 8);
  EXPECT_EQ(result.dominance[0].replacement_total, 9);
}

TEST_F(RegionSummaryTest, TiesAndUnknownAdaptersDoNotDeleteEveryState) {
  auto module = scalingDot();
  auto mesh = selectMesh(*module);
  auto type =
      mlir::RankedTensorType::get({8, 4}, mlir::Float32Type::get(&context));
  auto choices = tensorLayoutChoices(type, mesh, {}, {});
  for (bool unknown : {false, true}) {
    RegionSummary summary;
    summary.interface.inputs = {{0, type}};
    summary.interface.outputs = {{1, type}};
    summary.plans = {{0, {{choices[0]}, {choices[0]}}, {1, 0, 0}, ""},
                     {1, {{choices[1]}, {choices[1]}}, {1, 0, 0}, ""}};
    summary.finalizePlans();
    pruneDominatedStates(summary, [&](auto, auto, auto) {
      return Cost{0, 0, unknown ? 1U : 0U};
    });
    EXPECT_EQ(summary.frontier.size(), unknown ? 2 : 1);
  }
}

}  // namespace

}  // namespace joint_shard

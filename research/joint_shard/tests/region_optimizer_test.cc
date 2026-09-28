#include "research/joint_shard/search/region_optimizer.h"

#include <limits>
#include <set>

#include "research/joint_shard/reporting/reports.h"
#include "research/joint_shard/sharding/shardy_runner.h"
#include "research/joint_shard/tests/support/region_test.h"

namespace joint_shard {
namespace {
using RegionOptimizerTest = test::RegionTest;
TEST_F(RegionOptimizerTest, EndToEndBuildsFrontierWithoutMutatingSource) {
  auto module = scalingDot();
  auto before = print(*module);
  auto report = summarizeRegions(*module);
  ASSERT_EQ(report.regions.size(), 1);
  auto& summary = report.regions[0];
  EXPECT_GE(summary.candidates.size(), 2);
  EXPECT_EQ(summary.input_states_evaluated, 9);
  EXPECT_EQ(summary.evaluations, 9 * summary.candidates.size());
  EXPECT_LE(summary.best_boundary_plans, summary.feasible_plans);
  EXPECT_EQ(summary.unknown_cost_plans, 0);
  EXPECT_FALSE(summary.plans.empty());
  EXPECT_LT(summary.frontier.size(), summary.best_boundary_plans);
  EXPECT_EQ(summary.plans.size(), summary.best_boundary_plans);
  for (const auto& plan : summary.plans) {
    EXPECT_TRUE(plan.cost.known());
    if (std::find(summary.frontier.begin(), summary.frontier.end(), plan.id) !=
        summary.frontier.end())
      EXPECT_NE(plan.lowered_mlir.find("func.func @main"), std::string::npos);
    else
      EXPECT_TRUE(plan.lowered_mlir.empty());
  }
  EXPECT_EQ(print(*module), before);
  EXPECT_NE(formatReport(report).find("After reshard dominance:"),
            std::string::npos);
}

TEST_F(RegionOptimizerTest, ObserversReceiveCandidatesStagesAndFinalSummaries) {
  auto module = scalingDot();
  RegionOptimizerOptions options;
  options.max_candidates = 1;
  options.max_input_states = 1;
  size_t candidates = 0, snapshots = 0, summaries = 0;
  OptimizationObserver observer;
  observer.candidate_prepared = [&](size_t region, size_t candidate,
                                    mlir::ModuleOp prepared) {
    EXPECT_EQ(region, 0);
    EXPECT_EQ(candidate, 0);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(prepared)));
    ++candidates;
  };
  observer.snapshot = [&](size_t region, size_t boundary, size_t candidate,
                          const ShardySnapshot& snapshot) {
    EXPECT_EQ(region, 0);
    EXPECT_EQ(boundary, 0);
    EXPECT_EQ(candidate, 0);
    EXPECT_FALSE(snapshot.statistics);
    EXPECT_TRUE(parse(snapshot.mlir));
    ++snapshots;
  };
  observer.region_completed = [&](const RegionSummary& summary) {
    EXPECT_EQ(summary.id, 0);
    EXPECT_EQ(summary.plans.size(), 1);
    EXPECT_EQ(summary.frontier.size(), 1);
    ++summaries;
  };
  auto report = summarizeRegions(*module, options, observer);
  EXPECT_EQ(candidates, 1);
  EXPECT_EQ(snapshots, 4);
  EXPECT_EQ(summaries, 1);
  ASSERT_EQ(report.regions.size(), 1);
}

TEST_F(RegionOptimizerTest, ValidatesRatesAndPartitionCount) {
  EXPECT_THROW(CostModel(CostModelOptions{0, 1, 0}), std::invalid_argument);
  EXPECT_THROW(CostModel(CostModelOptions{1, 1, -1}), std::invalid_argument);
  auto module = scalingDot();
  module->getOperation()->setAttr(
      "mhlo.num_partitions",
      mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 32), 8));
  EXPECT_THROW(summarizeRegions(*module), std::invalid_argument);
}
}  // namespace

}  // namespace joint_shard

#include "research/joint_shard/sharding/shardy_runner.h"

#include <array>
#include <string>
#include <vector>

#include "research/joint_shard/tests/support/mlir_test.h"

namespace joint_shard {
namespace {
class ShardyRunnerTest : public test::MlirTest {
 protected:
  mlir::OwningOpRef<mlir::ModuleOp> annotatedDot() {
    return parse(R"mlir(module {
      sdy.mesh @mesh = <["data"=2, "model"=2]>
      func.func @main(
          %x: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {}]>},
          %w: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"model"}]>}
      ) -> (tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {}]>}) {
        %dot = stablehlo.dot_general %x, %w, contracting_dims = [1] x [0]
            : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
        return %dot : tensor<8x32xf32>
      }
    })mlir");
  }
};

TEST_F(ShardyRunnerTest, DefaultCaptureRetainsOnlyTheRequestedFinalStage) {
  const std::array stages = {ShardyStage::Propagation,
                             ShardyStage::ExplicitReshards,
                             ShardyStage::Collectives};
  const std::array names = {"01_propagated", "02_explicit_reshards",
                            "03_collectives"};
  ShardyRunner runner;
  for (size_t i = 0; i < stages.size(); ++i) {
    auto module = annotatedDot();
    ShardyRunOptions options;
    options.stop_after = stages[i];
    ASSERT_TRUE(mlir::succeeded(runner.run(*module, options)));
    ASSERT_EQ(runner.snapshots().size(), 1);
    const auto& snapshot = runner.snapshots().front();
    EXPECT_EQ(snapshot.name, names[i]);
    EXPECT_FALSE(snapshot.statistics);
    EXPECT_TRUE(parse(snapshot.mlir));
    const auto expected = snapshot.mlir;
    EXPECT_EQ(runner.takeFinalMlir(), expected);
  }
}

TEST_F(ShardyRunnerTest, AllStagesAndStatisticsPreserveTheSameFinalIR) {
  auto source = annotatedDot();
  mlir::OwningOpRef<mlir::ModuleOp> final_only(source->clone());
  ShardyRunner runner;
  ASSERT_TRUE(mlir::succeeded(runner.run(*final_only)));
  const auto expected = runner.takeFinalMlir();
  std::vector<std::string> observed;
  ShardyRunOptions options;
  options.capture = SnapshotCapture::AllStages;
  options.collect_statistics = true;
  options.on_snapshot = [&](const ShardySnapshot& snapshot) {
    EXPECT_TRUE(parse(snapshot.mlir));
    ASSERT_TRUE(snapshot.statistics);
    observed.push_back(snapshot.name);
  };
  ASSERT_TRUE(mlir::succeeded(runner.run(*source, options)));
  EXPECT_EQ(observed, (std::vector<std::string>{"00_input", "01_propagated",
                                                "02_explicit_reshards",
                                                "03_collectives"}));
  ASSERT_EQ(runner.snapshots().size(), 4);
  const auto& final = runner.snapshots().back();
  EXPECT_EQ(final.mlir, expected);
  ASSERT_TRUE(final.statistics);
  EXPECT_EQ(final.statistics->communication_ops.at("sdy.all_gather"), 1);
  EXPECT_GT(final.statistics->communication_payload_bytes, 0);
}
}  // namespace
}  // namespace joint_shard

#include "research/joint_shard/sharding/region_evaluator.h"

#include <limits>
#include <set>

#include "research/joint_shard/tests/support/region_test.h"
#include "research/joint_shard/transforms/region_candidates.h"

namespace joint_shard {
namespace {
using RegionEvaluatorTest = test::RegionTest;
TEST_F(RegionEvaluatorTest, EvaluationReachesShardyAndPreservesOriginalModule) {
  auto module = scalingDot();
  auto before = print(*module);
  auto region = regions(*module)[0];
  auto mesh = selectMesh(*module);
  TensorRewriteOptions options;
  auto saturated = saturateRegion(region, compileTensorRules(options), options);
  auto prepared = prepareCandidateModule(region, saturated.original, mesh);
  auto choicesA = tensorLayoutChoices(
      llvm::cast<mlir::RankedTensorType>(region.inputs[1].getType()), mesh, {},
      {});
  auto choicesB = tensorLayoutChoices(
      llvm::cast<mlir::RankedTensorType>(region.inputs[2].getType()), mesh, {},
      {});
  auto choicesOut = tensorLayoutChoices(
      llvm::cast<mlir::RankedTensorType>(region.outputs[0].getType()), mesh, {},
      {});
  BoundaryState state{{replicatedSharding(llvm::cast<mlir::RankedTensorType>(
                                              region.inputs[0].getType()),
                                          mesh),
                       choicesA[1], choicesB[2]},
                      {choicesOut[1]}};
  auto result = RegionEvaluator(mesh).evaluate(*prepared, state);
  ASSERT_TRUE(result.feasible) << result.failure;
  auto lowered = parse(result.lowered_mlir);
  size_t gathers = 0;
  lowered->walk([&](mlir::Operation* op) {
    if (op->getName().getStringRef() == "sdy.all_gather") ++gathers;
  });
  EXPECT_EQ(gathers, 1);
  EXPECT_TRUE(result.cost.known());
  EXPECT_GT(result.cost.communication, 5);
  EXPECT_EQ(print(*module), before);
  auto preparedFunction = prepared->lookupSymbol<mlir::func::FuncOp>("main");
  EXPECT_FALSE(preparedFunction.getArgAttr(0, "sdy.sharding"));
  state.inputs.pop_back();
  EXPECT_FALSE(RegionEvaluator(mesh).evaluate(*prepared, state).feasible);
}

TEST_F(RegionEvaluatorTest, OracleIsDirectionalAndCachesActualLowering) {
  auto module = scalingDot();
  auto mesh = selectMesh(*module);
  auto type =
      mlir::RankedTensorType::get({8, 4}, mlir::Float32Type::get(&context));
  auto choices = tensorLayoutChoices(type, mesh, {}, {});
  ReshardCostOracle oracle(mesh);
  auto slicing = oracle.estimate(choices[0], choices[1], type);
  auto gathering = oracle.estimate(choices[1], choices[0], type);
  ASSERT_TRUE(slicing.known());
  ASSERT_TRUE(gathering.known());
  EXPECT_EQ(slicing.total(), 0);
  EXPECT_GT(gathering.total(), 5);
  EXPECT_EQ(oracle.cacheSize(), 2);
  EXPECT_EQ(oracle.estimate(choices[1], choices[0], type).total(),
            gathering.total());
  EXPECT_EQ(oracle.cacheSize(), 2);
  EXPECT_EQ(oracle.estimate(choices[0], choices[0], type).total(), 0);
}

}  // namespace

}  // namespace joint_shard

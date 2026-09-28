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
  BoundaryState state{{replicatedSharding(llvm::cast<mlir::RankedTensorType>(
                                              region.inputs[0].getType()),
                                          mesh),
                       choicesA[1], choicesB[2]},
                      {}};
  auto result = RegionEvaluator(mesh).evaluate(*prepared, state.inputs);
  ASSERT_TRUE(result.feasible) << result.failure;
  auto lowered = parse(result.lowered_mlir);
  size_t gathers = 0;
  lowered->walk([&](mlir::Operation* op) {
    if (op->getName().getStringRef() == "sdy.all_gather") ++gathers;
  });
  EXPECT_EQ(gathers, 0);
  ASSERT_EQ(result.boundary.outputs.size(), 1);
  auto output = result.boundary.outputs[0].attr;
  EXPECT_EQ(output.getDimSharding(0).getAxes()[0].getName(), "data");
  EXPECT_EQ(output.getDimSharding(1).getAxes()[0].getName(), "model");
  EXPECT_EQ(result.boundary.inputs, state.inputs);
  EXPECT_TRUE(result.cost.known());
  EXPECT_EQ(result.cost.communication, 0);
  EXPECT_EQ(print(*module), before);
  auto preparedFunction = prepared->lookupSymbol<mlir::func::FuncOp>("main");
  EXPECT_FALSE(preparedFunction.getArgAttr(0, "sdy.sharding"));
  state.inputs.pop_back();
  EXPECT_FALSE(
      RegionEvaluator(mesh).evaluate(*prepared, state.inputs).feasible);
}

TEST_F(RegionEvaluatorTest,
       RequiredOutputContractIsPreservedWithoutEnumeration) {
  auto module = scalingDot();
  auto region = regions(*module)[0];
  auto mesh = selectMesh(*module);
  TensorRewriteOptions options;
  auto saturated = saturateRegion(region, compileTensorRules(options), options);
  auto output = tensorLayoutChoices(
      llvm::cast<mlir::RankedTensorType>(region.outputs[0].getType()), mesh, {},
      {})[1];
  auto prepared = prepareCandidateModule(
      region, saturated.original, mesh,
      {{region.outputs[0].getAsOpaquePointer(), output}});
  auto inputs = enumerateInputStates(region, mesh).states;
  auto result = RegionEvaluator(mesh).evaluate(*prepared, inputs[5]);
  ASSERT_TRUE(result.feasible) << result.failure;
  EXPECT_EQ(result.boundary.outputs, (std::vector<TensorSharding>{output}));
  EXPECT_GT(result.cost.communication, 0);
}

TEST_F(RegionEvaluatorTest, OpenOutputConstraintAllowsCombinedInference) {
  auto module = scalingDot();
  auto region = regions(*module)[0];
  auto mesh = selectMesh(*module);
  auto constraint = mlir::sdy::TensorShardingAttr::get(
      &context, mesh.name,
      {mlir::sdy::DimensionShardingAttr::get(
           &context, {mlir::sdy::AxisRefAttr::get(&context, "data")}, true),
       mlir::sdy::DimensionShardingAttr::get(&context, {}, false)},
      {}, {});
  module->lookupSymbol<mlir::func::FuncOp>("main").setResultAttr(
      0, "sdy.sharding", constraint);
  TensorRewriteOptions options;
  auto saturated = saturateRegion(region, compileTensorRules(options), options);
  auto prepared = prepareCandidateModule(region, saturated.original, mesh);
  auto inputs = enumerateInputStates(region, mesh).states;
  auto result = RegionEvaluator(mesh).evaluate(*prepared, inputs[5]);
  ASSERT_TRUE(result.feasible) << result.failure;
  auto output = result.boundary.outputs[0].attr;
  EXPECT_TRUE(output.isFullyClosed());
  EXPECT_EQ(output.getDimSharding(0).getAxes()[0].getName(), "data");
  ASSERT_EQ(output.getDimSharding(1).getAxes().size(), 1);
  EXPECT_EQ(output.getDimSharding(1).getAxes()[0].getName(), "model");
}

TEST_F(RegionEvaluatorTest, ReplicatedAndScalarOutputsGetExplicitContracts) {
  auto module = parse(R"mlir(module {
    sdy.mesh @mesh = <["data"=2, "model"=2]>
    func.func @main(%x: tensor<8xf32>) -> (tensor<8xf32>, tensor<f32>) {
      %y = stablehlo.negate %x : tensor<8xf32>
      %z = stablehlo.constant dense<0.0> : tensor<f32>
      return %y, %z : tensor<8xf32>, tensor<f32>
    }
  })mlir");
  auto mesh = selectMesh(*module);
  auto type =
      mlir::RankedTensorType::get({8}, mlir::Float32Type::get(&context));
  auto result =
      RegionEvaluator(mesh).evaluate(*module, {replicatedSharding(type, mesh)});
  ASSERT_TRUE(result.feasible) << result.failure;
  auto lowered = parse(result.lowered_mlir);
  auto fn = lowered->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_EQ(result.boundary.outputs.size(), 2);
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_EQ(
        result.boundary.outputs[i],
        replicatedSharding(
            llvm::cast<mlir::RankedTensorType>(fn.getResultTypes()[i]), mesh));
    EXPECT_EQ(fn.getResultAttr(i, "sdy.sharding"),
              result.boundary.outputs[i].attr);
  }
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

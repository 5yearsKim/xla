#include "research/joint_shard/sharding/boundary_state.h"

#include <limits>
#include <set>

#include "research/joint_shard/tests/support/region_test.h"

namespace joint_shard {
namespace {
using BoundaryStateTest = test::RegionTest;
TEST_F(BoundaryStateTest, BoundaryStatesAreExactTypedAndBounded) {
  auto module = scalingDot();
  auto region = regions(*module)[0];
  auto mesh = selectMesh(*module);
  auto all = enumerateBoundaryStates(region, mesh);
  EXPECT_EQ(all.states.size(), 27);  // scalar R; 3 choices for A, B, result.
  EXPECT_FALSE(all.truncated);
  for (const auto& state : all.states) {
    EXPECT_EQ(state.inputs[0].attr.getRank(), 0);
    for (const auto& input : state.inputs)
      EXPECT_TRUE(input.attr.isFullyClosed());
    for (const auto& output : state.outputs)
      EXPECT_TRUE(output.attr.isFullyClosed());
  }
  auto capped = enumerateBoundaryStates(region, mesh, {}, 5);
  EXPECT_EQ(capped.states.size(), 5);
  EXPECT_TRUE(capped.truncated);
  auto exactCap = enumerateBoundaryStates(region, mesh, {}, 27);
  EXPECT_FALSE(exactCap.truncated);
  auto vector =
      mlir::RankedTensorType::get({7}, mlir::Float32Type::get(&context));
  EXPECT_EQ(tensorLayoutChoices(vector, mesh, {}, {}).size(), 1);
}

TEST_F(BoundaryStateTest, FixedCombinedLayoutIsPreservedOutsideCanonicalSet) {
  auto module = parse(R"mlir(module {
    sdy.mesh @mesh = <["data"=2, "model"=2]>
    func.func @main(%x: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>})
        -> (tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>}) {
      %a = stablehlo.negate %x : tensor<8x4xf32>
      return %a : tensor<8x4xf32>
    }
  })mlir");
  auto states =
      enumerateBoundaryStates(regions(*module)[0], selectMesh(*module));
  ASSERT_EQ(states.states.size(), 1);
  EXPECT_EQ(states.states[0].inputs[0], states.states[0].outputs[0]);
  EXPECT_EQ(states.states[0].inputs[0].attr.getDimSharding(0).getAxes().size(),
            1);
  EXPECT_EQ(states.states[0].inputs[0].attr.getDimSharding(1).getAxes().size(),
            1);
}

TEST_F(BoundaryStateTest, PreservesDisjointShardingAndReplicatedSubAxes) {
  auto module = parse(R"mlir(module {
    sdy.mesh @mesh = <["model"=4]>
    func.func @main(%x: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model":(1)2}], replicated={"model":(2)2}>})
        -> tensor<8xf32> {
      %a = stablehlo.negate %x : tensor<8xf32>
      return %a : tensor<8xf32>
    }
  })mlir");
  auto mesh = selectMesh(*module);
  auto states = enumerateBoundaryStates(regions(*module)[0], mesh);
  ASSERT_FALSE(states.states.empty());
  for (const auto& state : states.states) {
    auto axes = state.inputs[0].attr.getDimSharding(0).getAxes();
    ASSERT_EQ(axes.size(), 1);
    EXPECT_EQ(axes[0].getSubAxisInfo().getSize(), 2);
  }
}

TEST_F(BoundaryStateTest, OpenConstraintsPermitOnlyCompatibleRefinements) {
  auto module = parse(R"mlir(module {
    sdy.mesh @mesh = <["data"=2, "model"=2]>
    func.func @main(%x: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {?}]>}) -> tensor<8x4xf32> {
      %a = stablehlo.negate %x : tensor<8x4xf32>
      return %a : tensor<8x4xf32>
    }
  })mlir");
  auto states =
      enumerateBoundaryStates(regions(*module)[0], selectMesh(*module));
  ASSERT_EQ(states.states.size(), 6);
  for (const auto& state : states.states)
    EXPECT_TRUE(state.inputs[0].attr.getDimSharding(0).getAxes().empty());
}

}  // namespace

}  // namespace joint_shard

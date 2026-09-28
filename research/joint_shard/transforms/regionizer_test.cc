#include "research/joint_shard/transforms/regionizer.h"

#include "research/joint_shard/tests/support/region_test.h"

namespace joint_shard {
namespace {
using RegionizerTest = test::RegionTest;
TEST_F(RegionizerTest, UnsupportedOperationsSplitWithoutDuplication) {
  auto module = parse(R"mlir(module {
    func.func @main(%x: tensor<8xf32>) -> tensor<8xf32> {
      %a = stablehlo.negate %x : tensor<8xf32>
      %b = stablehlo.sine %a : tensor<8xf32>
      %c = stablehlo.negate %b : tensor<8xf32>
      return %c : tensor<8xf32>
    }
  })mlir");
  auto split = regions(*module);
  ASSERT_EQ(split.size(), 2);
  EXPECT_EQ(split[0].operations.size(), 1);
  EXPECT_EQ(split[1].operations.size(), 1);
  EXPECT_NE(split[0].operations[0], split[1].operations[0]);
  EXPECT_EQ(split[0].outputs[0],
            split[1].inputs[0].getDefiningOp()->getOperand(0));
}

TEST_F(RegionizerTest, SizeLimitAndCutUseDistinctCrossingValues) {
  auto module = parse(R"mlir(module {
    func.func @main(%x: tensor<8xf32>, %y: tensor<8xf32>) -> tensor<8xf32> {
      %a = stablehlo.add %x, %y : tensor<8xf32>
      %b = stablehlo.multiply %a, %x : tensor<8xf32>
      %c = stablehlo.subtract %a, %y : tensor<8xf32>
      %d = stablehlo.add %b, %c : tensor<8xf32>
      %e = stablehlo.multiply %d, %d : tensor<8xf32>
      %f = stablehlo.negate %e : tensor<8xf32>
      return %f : tensor<8xf32>
    }
  })mlir");
  auto whole = regions(*module);
  ASSERT_EQ(whole.size(), 1);
  EXPECT_EQ(countCrossingValues(whole[0].operations, 2), 2);
  EXPECT_EQ(countCrossingValues(whole[0].operations, 3), 2);
  EXPECT_EQ(countCrossingValues(whole[0].operations, 4), 1);
  auto split = regions(*module, 4);
  ASSERT_EQ(split.size(), 2);
  EXPECT_EQ(split[0].operations.size(), 4);  // Cheaper cut beats midpoint.
  EXPECT_EQ(split[1].operations.size(), 2);
  auto bounded = regions(*module, 2);
  size_t total = 0;
  for (const auto& region : bounded) {
    EXPECT_LE(region.operations.size(), 2);
    total += region.operations.size();
  }
  EXPECT_EQ(total, 6);
}

TEST_F(RegionizerTest, ProtectsNonAdjacentTransposeDotAndReportsOversize) {
  auto module = parse(R"mlir(module {
    func.func @main(%a: tensor<8x16xf32>, %b: tensor<4x16xf32>) -> tensor<8x4xf32> {
      %t = stablehlo.transpose %b, dims = [1, 0] : (tensor<4x16xf32>) -> tensor<16x4xf32>
      %unused = stablehlo.negate %a : tensor<8x16xf32>
      %dot = stablehlo.dot_general %a, %t, contracting_dims = [1] x [0]
          : (tensor<8x16xf32>, tensor<16x4xf32>) -> tensor<8x4xf32>
      return %dot : tensor<8x4xf32>
    }
  })mlir");
  auto split = regions(*module, 1);
  ASSERT_EQ(split.size(), 1);
  EXPECT_TRUE(split[0].oversized);
  EXPECT_TRUE(cutBreaksProtectedPattern(split[0].operations, 1));
  EXPECT_TRUE(cutBreaksProtectedPattern(split[0].operations, 2));
}

TEST_F(RegionizerTest, ProtectsReshapesAndBroadcastMultiplyDot) {
  auto module = scalingDot();
  auto split = regions(*module, 1);
  ASSERT_EQ(split.size(), 1);
  EXPECT_TRUE(split[0].oversized);
  auto reshapes = parse(R"mlir(module {
    func.func @main(%a: tensor<8x4xf32>) -> tensor<8x4xf32> {
      %r = stablehlo.reshape %a : (tensor<8x4xf32>) -> tensor<32xf32>
      %s = stablehlo.reshape %r : (tensor<32xf32>) -> tensor<8x4xf32>
      return %s : tensor<8x4xf32>
    }
  })mlir");
  EXPECT_EQ(regions(*reshapes, 1).size(), 1);
}

TEST_F(RegionizerTest, InterfacesIncludeSharedValuesAndFunctionReturns) {
  auto module = parse(R"mlir(module {
    func.func @main(%x: tensor<8xf32>, %y: tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
      %a = stablehlo.add %x, %y : tensor<8xf32>
      %b = stablehlo.multiply %a, %a : tensor<8xf32>
      %c = stablehlo.negate %b : tensor<8xf32>
      return %a, %c : tensor<8xf32>, tensor<8xf32>
    }
  })mlir");
  auto whole = regions(*module)[0];
  auto first = describeRegion(llvm::ArrayRef(whole.operations).take_front(2));
  ASSERT_EQ(first.inputs.size(), 2);
  ASSERT_EQ(first.outputs.size(), 2);
  EXPECT_EQ(first.outputs[0], whole.operations[0]->getResult(0));
  EXPECT_EQ(first.outputs[1], whole.operations[1]->getResult(0));
  auto last = describeRegion(llvm::ArrayRef(whole.operations).drop_front(2));
  ASSERT_EQ(last.inputs.size(), 1);
  EXPECT_EQ(last.inputs[0], first.outputs[1]);
}

TEST_F(RegionizerTest, SupportedRuleBlockersRemainEvaluatedSingletons) {
  auto module = parse(R"mlir(module {
    func.func @main(%x: tensor<8xf32>) -> tensor<8xf32> {
      %a = stablehlo.negate %x : tensor<8xf32>
      %b = stablehlo.tanh %a : tensor<8xf32>
      %c = stablehlo.negate %b : tensor<8xf32>
      return %c : tensor<8xf32>
    }
  })mlir");
  auto function = module->lookupSymbol<mlir::func::FuncOp>("main");
  EXPECT_EQ(Regionizer().split(function).size(), 3);
  EXPECT_EQ(regions(*module).size(), 1);
}

}  // namespace

}  // namespace joint_shard

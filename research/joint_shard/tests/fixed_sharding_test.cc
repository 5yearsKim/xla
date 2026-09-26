#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"
#include "research/joint_shard/sharding/shardy_runner.h"
#include "research/joint_shard/transforms/rewrite_regions.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/register.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace {

class FixedShardingTest : public ::testing::Test {
 protected:
  FixedShardingTest() {
    mlir::DialectRegistry registry;
    mlir::stablehlo::registerAllDialects(registry);
    mlir::sdy::registerAllDialects(registry);
    context_.appendDialectRegistry(registry);
  }
  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef text) {
    return mlir::parseSourceString<mlir::ModuleOp>(text, &context_);
  }
  mlir::MLIRContext context_;
};

TEST_F(FixedShardingTest, RewritesBothSidesWithoutCrossingAnnotatedOperation) {
  auto module = parse(R"mlir(
module {
  sdy.mesh @custom = <["model"=2]>
  func.func @main(%a: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@custom, [{}, {}]>},
                  %b: tensor<8x8xf32>, %c: tensor<8x8xf32>)
      -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@custom, [{}, {}]>}) {
    %m = stablehlo.multiply %a, %b : tensor<8x8xf32>
    %before = stablehlo.add %m, %c : tensor<8x8xf32>
    %fixed = stablehlo.add %before, %c {sdy.sharding = #sdy.sharding_per_value<[<@custom, [{"model"}, {?}]>]>} : tensor<8x8xf32>
    %after = stablehlo.multiply %fixed, %b : tensor<8x8xf32>
    %out = stablehlo.add %after, %c : tensor<8x8xf32>
    return %out : tensor<8x8xf32>
  }
})mlir");
  ASSERT_TRUE(module);
  auto function = module->lookupSymbol<mlir::func::FuncOp>("main");
  auto argAttrs = function.getArgAttrDict(0);
  auto resultAttrs = function.getResultAttrDict(0);
  mlir::stablehlo::AddOp fixed;
  module->walk([&](mlir::stablehlo::AddOp op) {
    if (op->hasAttr("sdy.sharding")) fixed = op;
  });
  ASSERT_TRUE(fixed);
  auto attrs = fixed->getAttrDictionary();
  ASSERT_TRUE(mlir::succeeded(rewriteUnconstrainedRegions(*module)));
  EXPECT_EQ(fixed->getAttrDictionary(), attrs);
  EXPECT_EQ(fixed.getRhs(), function.getArgument(2));
  auto before = fixed.getLhs().getDefiningOp<mlir::stablehlo::AddOp>();
  ASSERT_TRUE(before);
  EXPECT_EQ(before.getLhs(), function.getArgument(2));
  EXPECT_TRUE(before.getRhs().getDefiningOp<mlir::stablehlo::MulOp>());
  auto ret = llvm::cast<mlir::func::ReturnOp>(
      function.getBody().front().getTerminator());
  auto after = ret.getOperand(0).getDefiningOp<mlir::stablehlo::AddOp>();
  ASSERT_TRUE(after);
  EXPECT_EQ(after.getLhs(), function.getArgument(2));
  auto multiply = after.getRhs().getDefiningOp<mlir::stablehlo::MulOp>();
  ASSERT_TRUE(multiply);
  EXPECT_EQ(multiply.getLhs(), fixed.getResult());
  EXPECT_EQ(function.getArgAttrDict(0), argAttrs);
  EXPECT_EQ(function.getResultAttrDict(0), resultAttrs);
}

TEST_F(FixedShardingTest, PreservesUseScopedAndDanglingConstraints) {
  auto module = parse(R"mlir(
module {
  sdy.mesh @mesh = <["x"=2]>
  func.func @main(%a: tensor<8x8xf32>, %b: tensor<8x8xf32>, %c: tensor<8x8xf32>) -> (tensor<8x8xf32>, tensor<8x8xf32>) {
    %m = stablehlo.multiply %a, %b : tensor<8x8xf32>
    %v = stablehlo.add %m, %c : tensor<8x8xf32>
    %constrained = sdy.sharding_constraint %v <@mesh, [{"x"}, {}]> : tensor<8x8xf32>
    %dangling = sdy.sharding_constraint %v <@mesh, [{"x"}, {}]> : tensor<8x8xf32>
    %left = stablehlo.multiply %constrained, %b : tensor<8x8xf32>
    %right = stablehlo.multiply %v, %b : tensor<8x8xf32>
    return %left, %right : tensor<8x8xf32>, tensor<8x8xf32>
  }
})mlir");
  ASSERT_TRUE(module);
  auto function = module->lookupSymbol<mlir::func::FuncOp>("main");
  llvm::SmallVector<mlir::sdy::ShardingConstraintOp> constraints;
  module->walk(
      [&](mlir::sdy::ShardingConstraintOp op) { constraints.push_back(op); });
  ASSERT_EQ(constraints.size(), 2);
  auto attrs = constraints[0]->getAttrDictionary();
  ASSERT_TRUE(mlir::succeeded(rewriteUnconstrainedRegions(*module)));
  EXPECT_EQ(constraints[0]->getAttrDictionary(), attrs);
  EXPECT_TRUE(constraints[1]->use_empty());
  auto ret = llvm::cast<mlir::func::ReturnOp>(
      function.getBody().front().getTerminator());
  auto left = ret.getOperand(0).getDefiningOp<mlir::stablehlo::MulOp>();
  auto right = ret.getOperand(1).getDefiningOp<mlir::stablehlo::MulOp>();
  ASSERT_TRUE(left && right);
  EXPECT_EQ(left.getLhs(), constraints[0].getResult());
  EXPECT_TRUE(right.getLhs().getDefiningOp<mlir::stablehlo::AddOp>());
  EXPECT_NE(right.getLhs(), constraints[0].getResult());
}

TEST_F(FixedShardingTest, KeepsUnsupportedAndUnusedAnnotatedOperations) {
  auto module = parse(R"mlir(
module {
  sdy.mesh @mesh = <["x"=2]>
  func.func @main(%a: tensor<8x8xf32>, %b: tensor<8x8xf32>, %c: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %unused = stablehlo.add %a, %b {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : tensor<8x8xf32>
    %m = stablehlo.multiply %a, %b : tensor<8x8xf32>
    %v = stablehlo.add %m, %c : tensor<8x8xf32>
    %t = stablehlo.transpose %v, dims = [1, 0] : (tensor<8x8xf32>) -> tensor<8x8xf32>
    return %t : tensor<8x8xf32>
  }
})mlir");
  ASSERT_TRUE(module);
  mlir::Operation* unused = nullptr;
  mlir::stablehlo::TransposeOp transpose;
  module->walk([&](mlir::Operation* op) {
    if (op->hasAttr("sdy.sharding")) unused = op;
    if (auto t = llvm::dyn_cast<mlir::stablehlo::TransposeOp>(op))
      transpose = t;
  });
  ASSERT_TRUE(unused && transpose);
  auto attrs = unused->getAttrDictionary();
  ASSERT_TRUE(mlir::succeeded(rewriteUnconstrainedRegions(*module)));
  EXPECT_EQ(unused->getAttrDictionary(), attrs);
  EXPECT_TRUE(unused->use_empty());
  auto add = transpose.getOperand().getDefiningOp<mlir::stablehlo::AddOp>();
  ASSERT_TRUE(add);
  EXPECT_TRUE(llvm::isa<mlir::BlockArgument>(add.getLhs()));
}

TEST_F(FixedShardingTest, ReshardsConflictingLayoutsUsingExistingMesh) {
  auto module = parse(R"mlir(
module attributes {mhlo.num_partitions = 2 : i32} {
  sdy.mesh @custom = <["model"=2]>
  func.func @main(%a: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@custom, [{}, {}]>},
                  %b: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@custom, [{}, {"model"}]>})
      -> (tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@custom, [{}, {}]>}) {
    %v = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
    return %v : tensor<8x32xf32>
  }
})mlir");
  ASSERT_TRUE(module);
  auto function = module->lookupSymbol<mlir::func::FuncOp>("main");
  auto argAttrs = function.getArgAttrDict(1);
  auto resultAttrs = function.getResultAttrDict(0);
  ShardyRunner runner;
  ASSERT_TRUE(mlir::succeeded(runner.run(*module)));
  function = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(function);
  EXPECT_EQ(function.getArgAttrDict(1), argAttrs);
  EXPECT_EQ(function.getResultAttrDict(0), resultAttrs);
  EXPECT_TRUE(module->lookupSymbol<mlir::sdy::MeshOp>("custom"));
  EXPECT_GT(runner.snapshots().back().communicationOps.at("sdy.all_gather"), 0);
}

TEST_F(FixedShardingTest, ConvertsUseScopedConstraintToCommunication) {
  auto module = parse(R"mlir(
module {
  sdy.mesh @mesh = <["x"=2]>
  func.func @main(%a: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
      -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
    %c = sdy.sharding_constraint %a <@mesh, [{}, {}]> : tensor<8x8xf32>
    return %c, %a : tensor<8x8xf32>, tensor<8x8xf32>
  }
})mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(rewriteUnconstrainedRegions(*module)));
  ShardyRunner runner;
  ASSERT_TRUE(mlir::succeeded(runner.run(*module)));
  EXPECT_GT(runner.snapshots().back().communicationOps.at("sdy.all_gather"), 0);
  module->walk([](mlir::sdy::ShardingConstraintOp) {
    ADD_FAILURE() << "unlowered constraint";
  });
}

TEST_F(FixedShardingTest, RejectsPartitionCountMismatch) {
  auto module = parse(R"mlir(
module attributes {mhlo.num_partitions = 4 : i32} {
  sdy.mesh @mesh = <["x"=2]>
  func.func @main(%a: tensor<8xf32>) -> tensor<8xf32> { return %a : tensor<8xf32> }
})mlir");
  ASSERT_TRUE(module);
  ShardyRunner runner;
  EXPECT_TRUE(mlir::failed(runner.run(*module)));
}

}  // namespace

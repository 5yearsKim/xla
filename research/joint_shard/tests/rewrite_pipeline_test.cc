#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"
#include "research/joint_shard/bridge/stablehlo_importer.h"
#include "research/joint_shard/sharding/shardy_runner.h"
#include "research/joint_shard/transforms/rewrite_regions.h"
#include "shardy/dialect/sdy/ir/register.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace {
class RewritePipelineTest : public ::testing::Test {
 protected:
  mlir::MLIRContext context;
  RewritePipelineTest() {
    mlir::DialectRegistry registry;
    mlir::stablehlo::registerAllDialects(registry);
    mlir::sdy::registerAllDialects(registry);
    context.appendDialectRegistry(registry);
  }
  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef text) {
    return mlir::parseSourceString<mlir::ModuleOp>(text, &context);
  }
};
TEST_F(RewritePipelineTest, MultipleOutputsKeepSharedProducer) {
  auto module = parse(R"mlir(module {
    func.func @main(%x: tensor<2x3xf32>, %y: tensor<2x3xf32>) -> (tensor<2x3xf32>, tensor<2x3xf32>) {
      %a = stablehlo.add %x, %y : tensor<2x3xf32>
      %b = stablehlo.exponential %a : tensor<2x3xf32>
      return %a, %b : tensor<2x3xf32>, tensor<2x3xf32>
    }
  })mlir");
  ASSERT_TRUE(module);
  TensorRewriteReport report;
  ASSERT_TRUE(
      mlir::succeeded(rewriteUnconstrainedRegions(*module, {}, &report)));
  EXPECT_EQ(report.regions, 1);
  EXPECT_EQ(report.roots, 2);
  ASSERT_EQ(report.extractions.size(), 1);
  EXPECT_FALSE(report.extractions.front().attempted_dag);
  EXPECT_EQ(report.extractions.front().fallback, "multiple-outputs");
  auto function = module->lookupSymbol<mlir::func::FuncOp>("main");
  auto ret = llvm::cast<mlir::func::ReturnOp>(
      function.getBody().front().getTerminator());
  auto exp = ret.getOperand(1).getDefiningOp<mlir::stablehlo::ExpOp>();
  ASSERT_TRUE(exp);
  EXPECT_EQ(exp.getOperand(), ret.getOperand(0));
  unsigned adds = 0;
  module->walk([&](mlir::stablehlo::AddOp) { ++adds; });
  EXPECT_EQ(adds, 1);
}
TEST_F(RewritePipelineTest, SingleOutputDagAndBudgetFallbackPreserveSharing) {
  for (bool limited : {false, true}) {
    auto module = parse(R"mlir(module {
      func.func @main(%a: tensor<2x4xi32>, %b: tensor<4x1xi32>) -> tensor<2x1xi32> {
        %dot = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0]
            : (tensor<2x4xi32>, tensor<4x1xi32>) -> tensor<2x1xi32>
        %sum = stablehlo.add %dot, %dot : tensor<2x1xi32>
        return %sum : tensor<2x1xi32>
      }
    })mlir");
    ASSERT_TRUE(module);
    TensorRewriteOptions options;
    // Keep this test focused on extraction and export of the imported DAG.
    options.runner.iteration_limit = 0;
    if (limited) options.dag.state_limit = 1;
    TensorRewriteReport report;
    ASSERT_TRUE(mlir::succeeded(
        rewriteUnconstrainedRegions(*module, options, &report)));
    ASSERT_EQ(report.extractions.size(), 1);
    const auto& extraction = report.extractions.front();
    EXPECT_TRUE(extraction.attempted_dag);
    EXPECT_EQ(extraction.fallback,
              limited ? "no-dag-candidate" : "baseline-no-worse");
    EXPECT_NE(report.str().find("selected=tree roots=1 attempted_dag=1"),
              std::string::npos);
    if (limited)
      EXPECT_NE(report.str().find("stop=state-limit"), std::string::npos);
    auto function = module->lookupSymbol<mlir::func::FuncOp>("main");
    auto ret = llvm::cast<mlir::func::ReturnOp>(
        function.getBody().front().getTerminator());
    auto sum = ret.getOperand(0).getDefiningOp<mlir::stablehlo::AddOp>();
    ASSERT_TRUE(sum);
    EXPECT_EQ(sum.getLhs(), sum.getRhs());
    unsigned dots = 0;
    module->walk([&](mlir::stablehlo::DotGeneralOp) { ++dots; });
    EXPECT_EQ(dots, 1);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
  }
}
TEST_F(RewritePipelineTest, DagSelectsAndExportsSharedDotAfterSaturation) {
  for (auto mode : {TensorExtractorMode::Auto, TensorExtractorMode::Tree}) {
    auto module = parse(R"mlir(module {
      func.func @main(%a: tensor<2x4xi32>, %b: tensor<4x1xi32>) -> tensor<2x1xi32> {
        %twice = stablehlo.add %a, %a : tensor<2x4xi32>
        %dot = stablehlo.dot_general %twice, %b, contracting_dims = [1] x [0]
            : (tensor<2x4xi32>, tensor<4x1xi32>) -> tensor<2x1xi32>
        return %dot : tensor<2x1xi32>
      }
    })mlir");
    ASSERT_TRUE(module);
    TensorRewriteOptions options;
    options.extractor = mode;
    options.semantic.enable_associativity = false;
    options.dag.time_limit.reset();
    TensorRewriteReport report;
    ASSERT_TRUE(mlir::succeeded(
        rewriteUnconstrainedRegions(*module, options, &report)));
    ASSERT_EQ(report.extractions.size(), 1);
    const auto& extraction = report.extractions.front();
    // Saturation can also double the smaller RHS (cost 22), while the
    // shared-dot form still costs only 20.
    EXPECT_EQ(extraction.baseline_cost, 22.0);
    auto function = module->lookupSymbol<mlir::func::FuncOp>("main");
    auto ret = llvm::cast<mlir::func::ReturnOp>(
        function.getBody().front().getTerminator());
    if (mode == TensorExtractorMode::Auto) {
      EXPECT_TRUE(extraction.selected_dag);
      EXPECT_EQ(extraction.candidate_cost, 20.0);
      auto sum = ret.getOperand(0).getDefiningOp<mlir::stablehlo::AddOp>();
      ASSERT_TRUE(sum);
      EXPECT_EQ(sum.getLhs(), sum.getRhs());
    } else {
      EXPECT_FALSE(extraction.attempted_dag);
      EXPECT_TRUE(
          ret.getOperand(0).getDefiningOp<mlir::stablehlo::DotGeneralOp>());
    }
    unsigned dots = 0;
    module->walk([&](mlir::stablehlo::DotGeneralOp) { ++dots; });
    EXPECT_EQ(dots, 1);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
  }
}
TEST_F(RewritePipelineTest, ScalarDotScalingExportsCorrectOutputBroadcast) {
  for (bool relaxed : {false, true}) {
    auto module = parse(R"mlir(module {
      func.func @main(%s: tensor<f32>, %a: tensor<2x3xf32>, %b: tensor<3x1xf32>) -> tensor<2x1xf32> {
        %scale = stablehlo.broadcast_in_dim %s, dims = []
            : (tensor<f32>) -> tensor<2x3xf32>
        %scaled = stablehlo.multiply %scale, %a : tensor<2x3xf32>
        %dot = stablehlo.dot_general %scaled, %b, contracting_dims = [1] x [0]
            : (tensor<2x3xf32>, tensor<3x1xf32>) -> tensor<2x1xf32>
        return %dot : tensor<2x1xf32>
      }
    })mlir");
    ASSERT_TRUE(module);
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
    TensorRewriteOptions options;
    options.numerical_policy = relaxed ? NumericalPolicy::AllowReassociation
                                       : NumericalPolicy::PreserveEvaluation;
    options.extraction = ExtractionProfile::Compute;
    ASSERT_TRUE(mlir::succeeded(rewriteUnconstrainedRegions(*module, options)));
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
    auto function = module->lookupSymbol<mlir::func::FuncOp>("main");
    auto ret = llvm::cast<mlir::func::ReturnOp>(
        function.getBody().front().getTerminator());
    auto* root = ret.getOperand(0).getDefiningOp();
    ASSERT_NE(root, nullptr);
    if (relaxed) {
      EXPECT_TRUE(llvm::isa<mlir::stablehlo::MulOp>(root));
      unsigned broadcasts = 0;
      module->walk([&](mlir::stablehlo::BroadcastInDimOp op) {
        ++broadcasts;
        EXPECT_EQ(op->getResult(0).getType(), ret.getOperand(0).getType());
        EXPECT_EQ(op->getOperand(0), function.getArgument(0));
      });
      EXPECT_EQ(broadcasts, 1);
      module->walk([&](mlir::stablehlo::DotGeneralOp op) {
        EXPECT_EQ(op->getOperand(0), function.getArgument(1));
        EXPECT_EQ(op->getOperand(1), function.getArgument(2));
      });
    } else {
      EXPECT_TRUE(llvm::isa<mlir::stablehlo::DotGeneralOp>(root));
      module->walk([&](mlir::stablehlo::BroadcastInDimOp op) {
        EXPECT_EQ(op->getResult(0).getType(),
                  function.getArgument(1).getType());
      });
    }
  }
}
TEST_F(RewritePipelineTest, BatchedDotAndReductionRoundTrip) {
  auto module = parse(R"mlir(module {
    func.func @main(%a: tensor<2x3x4xf32>, %b: tensor<2x4x5xf32>) -> tensor<2x3xf32> {
      %dot = stablehlo.dot_general %a, %b, batching_dims = [0] x [0],
          contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT]
          : (tensor<2x3x4xf32>, tensor<2x4x5xf32>) -> tensor<2x3x5xf32>
      %zero = sdy.constant dense<0.0> : tensor<f32>
      %sum = stablehlo.reduce(%dot init: %zero) applies stablehlo.add across dimensions = [2]
          : (tensor<2x3x5xf32>, tensor<f32>) -> tensor<2x3xf32>
      return %sum : tensor<2x3xf32>
    }
  })mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  unsigned admitted = 0;
  module->walk([&](mlir::Operation* op) {
    if (canImportTensorOperation(op)) ++admitted;
  });
  EXPECT_GE(admitted, 3);
  ASSERT_TRUE(mlir::succeeded(rewriteUnconstrainedRegions(*module)));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
  unsigned dots = 0, reductions = 0;
  module->walk([&](mlir::stablehlo::DotGeneralOp op) {
    ++dots;
    EXPECT_EQ(op.getDotDimensionNumbers().getLhsBatchingDimensions().size(), 1);
  });
  module->walk([&](mlir::stablehlo::ReduceOp) { ++reductions; });
  EXPECT_EQ(dots, 1);
  EXPECT_EQ(reductions, 1);
}
TEST_F(RewritePipelineTest, NonIdentityReduceAndUnknownMetadataAreBoundaries) {
  auto module = parse(R"mlir(module {
    func.func @main(%x: tensor<2x3xf32>) -> tensor<2xf32> {
      %one = stablehlo.constant dense<1.0> : tensor<f32>
      %x2 = stablehlo.add %x, %x {unknown_hint = "keep"} : tensor<2x3xf32>
      %r = stablehlo.reduce(%x2 init: %one) applies stablehlo.add across dimensions = [1]
          : (tensor<2x3xf32>, tensor<f32>) -> tensor<2xf32>
      return %r : tensor<2xf32>
    }
  })mlir");
  ASSERT_TRUE(module);
  mlir::Operation* originalAdd = nullptr;
  mlir::Operation* originalReduce = nullptr;
  module->walk([&](mlir::stablehlo::AddOp op) {
    if (op->hasAttr("unknown_hint")) originalAdd = op;
  });
  module->walk([&](mlir::stablehlo::ReduceOp op) { originalReduce = op; });
  ASSERT_NE(originalAdd, nullptr);
  ASSERT_NE(originalReduce, nullptr);
  EXPECT_FALSE(canImportTensorOperation(originalAdd));
  EXPECT_FALSE(canImportTensorOperation(originalReduce));
  auto attrs = originalAdd->getAttrDictionary();
  ASSERT_TRUE(mlir::succeeded(rewriteUnconstrainedRegions(*module)));
  EXPECT_EQ(originalAdd->getAttrDictionary(), attrs);
  EXPECT_EQ(originalReduce->getOperand(0).getDefiningOp(), originalAdd);
}
TEST_F(RewritePipelineTest, AbsorbsTransposeIntoBatchedDotDimensions) {
  auto module = parse(R"mlir(module {
    func.func @main(%a: tensor<2x3x4xf32>, %b: tensor<2x5x4xf32>) -> tensor<2x3x5xf32> {
      %t = stablehlo.transpose %b, dims = [0, 2, 1]
          {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "derived"}
          : (tensor<2x5x4xf32>) -> tensor<2x4x5xf32>
      %dot = stablehlo.dot_general %a, %t, batching_dims = [0] x [0], contracting_dims = [2] x [1]
          : (tensor<2x3x4xf32>, tensor<2x4x5xf32>) -> tensor<2x3x5xf32>
      return %dot : tensor<2x3x5xf32>
    }
  })mlir");
  ASSERT_TRUE(module);
  TensorRewriteOptions options;
  options.extraction = ExtractionProfile::Memory;
  ASSERT_TRUE(mlir::succeeded(rewriteUnconstrainedRegions(*module, options)));
  unsigned transposes = 0;
  module->walk([&](mlir::stablehlo::TransposeOp) { ++transposes; });
  EXPECT_EQ(transposes, 0);
  module->walk([&](mlir::stablehlo::DotGeneralOp op) {
    EXPECT_EQ(op.getDotDimensionNumbers().getRhsContractingDimensions()[0], 2);
  });
}
TEST_F(RewritePipelineTest, EmptyReductionRetainsCanonicalInitializer) {
  auto module = parse(R"mlir(module {
    func.func @main(%x: tensor<2x0xf32>) -> tensor<2xf32> {
      %zero = stablehlo.constant dense<0.0> : tensor<f32>
      %r = stablehlo.reduce(%x init: %zero) applies stablehlo.add across dimensions = [1]
          : (tensor<2x0xf32>, tensor<f32>) -> tensor<2xf32>
      return %r : tensor<2xf32>
    }
  })mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(rewriteUnconstrainedRegions(*module)));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
  module->walk([&](mlir::stablehlo::ReduceOp op) {
    auto init = op->getOperand(1).getDefiningOp();
    ASSERT_NE(init, nullptr);
    EXPECT_EQ(init->getAttrOfType<mlir::DenseElementsAttr>("value")
                  .getSplatValue<float>(),
              0.0f);
  });
}
TEST_F(RewritePipelineTest, CandidateCostKeepsBaselineOnTiesAndUnknowns) {
  ModuleCost baseline{100, 50, 0};
  EXPECT_FALSE(betterModuleCost(baseline, baseline));
  EXPECT_TRUE(betterModuleCost({99, 100, 0}, baseline));
  EXPECT_TRUE(betterModuleCost({100, 49, 0}, baseline));
  EXPECT_FALSE(betterModuleCost({0, 0, 1}, baseline));
  EXPECT_FALSE(betterModuleCost({0, 0, 0}, {100, 50, 1}));
}
TEST_F(RewritePipelineTest, RewriteCliLimitsAndPolicyAreValidated) {
  TensorRewriteOptions options;
  EXPECT_EQ(options.extractor, TensorExtractorMode::Auto);
  EXPECT_EQ(options.dag.state_limit, 10000);
  EXPECT_EQ(options.dag.time_limit, std::chrono::milliseconds(50));
  EXPECT_EQ(options.dag.frontier_limit, 1000);
  EXPECT_TRUE(parseTensorRewriteOption("--extractor=tree", options));
  EXPECT_EQ(options.extractor, TensorExtractorMode::Tree);
  EXPECT_TRUE(parseTensorRewriteOption("--extractor=auto", options));
  EXPECT_EQ(options.extractor, TensorExtractorMode::Auto);
  EXPECT_TRUE(parseTensorRewriteOption("--dag-states=123", options));
  EXPECT_EQ(options.dag.state_limit, 123);
  EXPECT_TRUE(parseTensorRewriteOption("--dag-time-ms=7", options));
  EXPECT_EQ(options.dag.time_limit, std::chrono::milliseconds(7));
  EXPECT_TRUE(parseTensorRewriteOption("--dag-frontier=12", options));
  EXPECT_EQ(options.dag.frontier_limit, 12);
  for (auto argument :
       {"--extractor=dag", "--dag-states=0", "--dag-frontier=-1",
        "--dag-time-ms=bad", "--dag-time-ms=18446744073709551615"})
    EXPECT_THROW(parseTensorRewriteOption(argument, options),
                 std::invalid_argument);
  EXPECT_TRUE(parseTensorRewriteOption("--numerical-policy=relaxed", options));
  EXPECT_EQ(options.numerical_policy, NumericalPolicy::AllowReassociation);
  EXPECT_TRUE(parseTensorRewriteOption("--search-visits=123", options));
  EXPECT_EQ(options.semantic.visit_limit, 123);
  EXPECT_THROW(parseTensorRewriteOption("--iterations=0", options),
               std::invalid_argument);
  EXPECT_THROW(parseTensorRewriteOption("--numerical-policy=maybe", options),
               std::invalid_argument);
}
}  // namespace

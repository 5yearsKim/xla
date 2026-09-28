#include <stdexcept>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "eggc/egraph.hpp"
#include "eggc/extract.hpp"
#include "eggc/runner.hpp"
#include "gtest/gtest.h"
#include "research/joint_shard/bridge/stablehlo_exporter.h"
#include "research/joint_shard/bridge/stablehlo_importer.h"
#include "research/joint_shard/bridge/tensor_lang/tensor_analysis.h"
#include "research/joint_shard/bridge/tensor_lang/tensor_rewrites.h"
#include "research/joint_shard/bridge/tensor_lang/tensorlang.h"
#include "research/joint_shard/tests/support/mlir_test.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace joint_shard {

static_assert(eggc::Language<TensorNode>);
static_assert(eggc::AnalysisFor<TensorAnalysis, TensorNode>);

namespace {

using StableHloBridgeTest = test::MlirTest;

TEST_F(StableHloBridgeTest, TypedRoundTripPreservesDotAttributesAndSharing) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%a: tensor<2x3xf32>, %b: tensor<3x4xf32>) -> tensor<2x4xf32> {
    %dot = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0],
        precision = [HIGH, HIGH] : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    %sum = stablehlo.add %dot, %dot : tensor<2x4xf32>
    return %sum : tensor<2x4xf32>
  }
})mlir",
                                                        &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  auto function = module->lookupSymbol<mlir::func::FuncOp>("main");
  auto ret = llvm::cast<mlir::func::ReturnOp>(
      function.getBody().front().getTerminator());
  auto originalAdd = ret.getOperand(0).getDefiningOp<mlir::stablehlo::AddOp>();
  auto originalDot =
      originalAdd.getLhs().getDefiningOp<mlir::stablehlo::DotGeneralOp>();
  TensorEGraph graph(TensorAnalysis{});
  StableHloImporter importer(graph);
  std::vector<mlir::Value> inputs{function.getArgument(0),
                                  function.getArgument(1)};
  for (unsigned i = 0; i < inputs.size(); ++i) importer.bindValue(inputs[i], i);
  auto root = importer.importValue(ret.getOperand(0));
  eggc::run(graph, buildSemanticRules());
  const auto& addNode = graph.nodes(root).front();
  const auto& dotNode = graph.nodes(addNode.operands[0]).front();
  ASSERT_EQ(dotNode.op, OpKind::DotGeneral);
  const auto* attrs = std::get_if<DotGeneralAttrs>(&dotNode.attrs);
  ASSERT_NE(attrs, nullptr);
  EXPECT_EQ(attrs->lhs_contracting, std::vector<int64_t>({1}));
  EXPECT_EQ(attrs->rhs_contracting, std::vector<int64_t>({0}));
  EXPECT_EQ(attrs->precision_config, originalDot->getAttr("precision_config"));
  auto expression = TensorExtractor(graph).find_best(root).second;
  mlir::OpBuilder builder(ret);
  StableHloExporter exporter(builder, ret.getLoc(), inputs);
  auto output = exporter.exportExpr(expression, expression.root());
  ret->setOperand(0, output);
  auto rebuiltAdd = output.getDefiningOp<mlir::stablehlo::AddOp>();
  ASSERT_TRUE(rebuiltAdd);
  EXPECT_EQ(rebuiltAdd.getLhs(), rebuiltAdd.getRhs());
  auto rebuiltDot =
      rebuiltAdd.getLhs().getDefiningOp<mlir::stablehlo::DotGeneralOp>();
  ASSERT_TRUE(rebuiltDot);
  EXPECT_EQ(originalDot->getAttrDictionary(), rebuiltDot->getAttrDictionary());
  EXPECT_EQ(tensorFacts(graph, root).type, output.getType());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST_F(StableHloBridgeTest, ExportReconstructsEditedNativeDotDimensions) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %dot = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0],
        precision = [HIGH, HIGH]
        : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %dot : tensor<2x2xf32>
  }
})mlir",
                                                        &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  auto function = module->lookupSymbol<mlir::func::FuncOp>("main");
  auto ret = llvm::cast<mlir::func::ReturnOp>(
      function.getBody().front().getTerminator());
  auto original =
      ret.getOperand(0).getDefiningOp<mlir::stablehlo::DotGeneralOp>();
  TensorEGraph graph(TensorAnalysis{});
  StableHloImporter importer(graph);
  std::vector<mlir::Value> inputs{function.getArgument(0),
                                  function.getArgument(1)};
  for (unsigned i = 0; i < inputs.size(); ++i) importer.bindValue(inputs[i], i);
  auto root = importer.importValue(ret.getOperand(0));
  graph.rebuild();
  auto expr = TensorExtractor(graph).find_best(root).second;
  auto& attrs = std::get<DotGeneralAttrs>(expr.nodes.back().attrs);
  attrs.lhs_contracting = {0};
  attrs.rhs_contracting = {1};
  mlir::OpBuilder builder(ret);
  StableHloExporter exporter(builder, ret.getLoc(), inputs);
  auto output = exporter.exportExpr(expr, expr.root());
  ret->setOperand(0, output);
  auto rebuilt = output.getDefiningOp<mlir::stablehlo::DotGeneralOp>();
  ASSERT_TRUE(rebuilt);
  EXPECT_NE(rebuilt.getDotDimensionNumbers(),
            original.getDotDimensionNumbers());
  EXPECT_EQ(rebuilt->getAttr("precision_config"),
            original->getAttr("precision_config"));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

}  // namespace

}  // namespace joint_shard

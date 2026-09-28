#include "research/joint_shard/export/selected_export.h"

#include <stdexcept>
#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "gtest/gtest.h"
#include "research/joint_shard/sharding/shardy_runner.h"
#include "research/joint_shard/tests/support/mlir_test.h"

namespace joint_shard {
namespace {

class SelectedExportTest : public test::MlirTest {
 protected:
  mlir::OwningOpRef<mlir::ModuleOp> fixture(const std::string& name) {
    auto module = mlir::parseSourceFile<mlir::ModuleOp>(
        "research/joint_shard/testdata/" + name + ".mlir", &context);
    if (!module) throw std::runtime_error("missing lowering fixture");
    return module;
  }

  size_t count(mlir::ModuleOp module, llvm::StringRef name) {
    size_t result = 0;
    module.walk([&](mlir::Operation* op) {
      result += op->getName().getStringRef() == name;
    });
    return result;
  }
};

TEST_F(SelectedExportTest, PreservesSelectedCollectivesAndInput) {
  for (const auto& name : {"dot_general", "dot_gather", "dot_contracting"}) {
    SCOPED_TRACE(name);
    auto selected = fixture(name);
    ShardyRunner runner;
    ASSERT_TRUE(mlir::succeeded(runner.run(*selected)));
    const auto before = print(*selected);
    auto lowered = exportSelectedProgram(*selected);
    ASSERT_TRUE(lowered.ok()) << lowered.status();
    EXPECT_EQ(print(*selected), before);
    auto local = parse(lowered->local_mlir);
    EXPECT_EQ(count(*local, "stablehlo.all_reduce"),
              count(*selected, "sdy.all_reduce"));
    EXPECT_EQ(count(*local, "stablehlo.all_gather"),
              count(*selected, "sdy.all_gather"));
    EXPECT_EQ(count(*lowered->module, "sdy.manual_computation"), 0);
    EXPECT_EQ(count(*lowered->module, "sdy.mesh"), 0);
    EXPECT_EQ(count(*lowered->module, "mhlo.copy"), 0);
    EXPECT_EQ(lowered->partitions, 4);
    EXPECT_EQ(lowered->module->getOperation()
                  ->getAttrOfType<mlir::IntegerAttr>("mhlo.num_partitions")
                  .getInt(),
              4);
  }
}

TEST_F(SelectedExportTest, RejectsMissingAndInconsistentBoundary) {
  auto source = fixture("dot_contracting");
  source->lookupSymbol<mlir::func::FuncOp>("main").removeArgAttr(
      0, "sdy.sharding");
  EXPECT_FALSE(exportSelectedProgram(*source).ok());
  source = fixture("dot_contracting");
  source->getOperation()->setAttr(
      "mhlo.num_partitions",
      mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 64), 2));
  EXPECT_FALSE(exportSelectedProgram(*source).ok());
  source->getOperation()->removeAttr("mhlo.num_partitions");
  source->getOperation()->setAttr(
      "mhlo.num_replicas",
      mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 64), 2));
  EXPECT_FALSE(exportSelectedProgram(*source).ok());
}

TEST_F(SelectedExportTest, RejectsOpenAndNondivisibleLayouts) {
  auto source = fixture("dot_contracting");
  auto text = print(*source);
  auto start = text.find("{\"data\"}");
  ASSERT_NE(start, std::string::npos);
  text.replace(start, std::string("{\"data\"}").size(), "{\"data\", ?}");
  auto open = parse(text);
  EXPECT_FALSE(exportSelectedProgram(*open).ok());
  text = print(*source);
  while ((start = text.find("8x")) != std::string::npos)
    text.replace(start, 2, "7x");
  auto nondivisible = parse(text);
  EXPECT_FALSE(exportSelectedProgram(*nondivisible).ok());
}

TEST_F(SelectedExportTest, RejectsUnresolvedReshard) {
  auto source = parse(R"mlir(module {
    sdy.mesh @mesh = <["model"=2]>
    func.func @main(%x: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>})
        -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}]>}) {
      %0 = sdy.reshard %x <@mesh, [{"model"}]> : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
  })mlir");
  EXPECT_FALSE(exportSelectedProgram(*source).ok());
}

}  // namespace
}  // namespace joint_shard

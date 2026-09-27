#ifndef RESEARCH_JOINT_SHARD_TESTS_SUPPORT_MLIR_TEST_H_
#define RESEARCH_JOINT_SHARD_TESTS_SUPPORT_MLIR_TEST_H_
#include <stdexcept>
#include <string>

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"
#include "shardy/dialect/sdy/ir/register.h"
#include "stablehlo/dialect/Register.h"

namespace joint_shard {
namespace test {
class MlirTest : public ::testing::Test {
 protected:
  MlirTest() {
    mlir::DialectRegistry registry;
    mlir::stablehlo::registerAllDialects(registry);
    mlir::sdy::registerAllDialects(registry);
    context.appendDialectRegistry(registry);
  }
  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef source) {
    auto module = parseUnchecked(source);
    if (!module || mlir::failed(mlir::verify(*module)))
      throw std::runtime_error("invalid test fixture");
    return module;
  }
  mlir::OwningOpRef<mlir::ModuleOp> parseUnchecked(llvm::StringRef source) {
    return mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  }
  std::string print(mlir::ModuleOp module) {
    std::string text;
    llvm::raw_string_ostream out(text);
    module.print(out);
    return text;
  }
  mlir::MLIRContext context;
};
}  // namespace test
}  // namespace joint_shard

#endif

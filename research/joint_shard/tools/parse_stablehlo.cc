#include <exception>

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "research/joint_shard/transforms/rewrite_regions.h"
#include "shardy/dialect/sdy/ir/register.h"
#include "stablehlo/dialect/Register.h"

int main(int argc, char** argv) {
  if (argc != 2) {
    llvm::errs() << "Usage: parse_stablehlo <file.mlir>\n";
    return 1;
  }
  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  mlir::sdy::registerAllDialects(registry);
  mlir::MLIRContext context(registry);
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
  if (!module || mlir::failed(mlir::verify(*module))) return 1;
  try {
    if (mlir::failed(rewriteUnconstrainedRegions(*module))) return 1;
  } catch (const std::exception& error) {
    llvm::errs() << error.what() << "\n";
    return 1;
  }
  module->print(llvm::outs());
  llvm::outs() << "\n";
  return 0;
}

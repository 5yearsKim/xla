#include <exception>
#include <string>

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "research/joint_shard/transforms/rewrite_regions.h"
#include "shardy/dialect/sdy/ir/register.h"
#include "stablehlo/dialect/Register.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    llvm::errs() << "Usage: parse_stablehlo <file.mlir> "
                 << tensorRewriteOptionHelp() << "\n";
    return 1;
  }
  try {
    TensorRewriteOptions options;
    bool showReport = false;
    for (int i = 2; i < argc; ++i) {
      if (std::string_view(argv[i]) == "--rewrite-report")
        showReport = true;
      else if (!parseTensorRewriteOption(argv[i], options))
        throw std::invalid_argument("unknown option: " + std::string(argv[i]));
    }
    mlir::DialectRegistry registry;
    mlir::stablehlo::registerAllDialects(registry);
    mlir::sdy::registerAllDialects(registry);
    mlir::MLIRContext context(registry);
    auto module = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
    if (!module || mlir::failed(mlir::verify(*module))) return 1;
    TensorRewriteReport report;
    if (mlir::failed(rewriteUnconstrainedRegions(*module, options, &report)))
      return 1;
    if (showReport) llvm::errs() << report.str();
    module->print(llvm::outs());
    llvm::outs() << "\n";
  } catch (const std::exception& error) {
    llvm::errs() << error.what() << "\n";
    return 1;
  }
  return 0;
}

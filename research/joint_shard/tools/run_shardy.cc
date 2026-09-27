#include <stdexcept>
#include <string>

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "research/joint_shard/sharding/shardy_runner.h"
#include "shardy/dialect/sdy/ir/register.h"
#include "stablehlo/dialect/Register.h"

// Raw pipeline inspection. Rewrite/layout search belongs to summarize_regions.
int main(int argc, char** argv) {
  if (argc < 2) {
    llvm::errs() << "Usage: run_shardy <file.mlir> "
                    "[--stop-after=propagation|reshards|collectives] "
                    "[--dump-dir=directory]\n";
    return 1;
  }
  try {
    ShardyRunOptions options;
    for (int i = 2; i < argc; ++i) {
      std::string argument(argv[i]);
      if (argument.starts_with("--dump-dir="))
        options.dumpDirectory = argument.substr(11);
      else if (argument.starts_with("--stop-after=")) {
        auto stage = argument.substr(13);
        if (stage == "propagation")
          options.stopAfter = ShardyStage::Propagation;
        else if (stage == "reshards")
          options.stopAfter = ShardyStage::ExplicitReshards;
        else if (stage == "collectives")
          options.stopAfter = ShardyStage::Collectives;
        else
          throw std::invalid_argument("unknown stop stage: " + stage);
      } else
        throw std::invalid_argument("unknown option: " + argument);
    }
    mlir::DialectRegistry registry;
    mlir::stablehlo::registerAllDialects(registry);
    mlir::sdy::registerAllDialects(registry);
    mlir::MLIRContext context(registry);
    auto module = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
    if (!module || mlir::failed(mlir::verify(*module))) return 1;
    ShardyRunner runner;
    if (mlir::failed(runner.run(*module, options))) return 1;
    const auto& snapshot = runner.snapshots().back();
    llvm::errs() << "logical_payload_bytes="
                 << snapshot.cost.communication_payload_bytes
                 << " global_compute_work=" << snapshot.cost.compute_work
                 << " unknown_costs=" << snapshot.cost.unknown_costs << "\n";
    llvm::outs() << snapshot.mlir;
  } catch (const std::exception& error) {
    llvm::errs() << error.what() << "\n";
    return 1;
  }
  return 0;
}

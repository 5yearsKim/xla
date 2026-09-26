#include <stdexcept>
#include <string>

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "research/joint_shard/sharding/shardy_runner.h"
#include "research/joint_shard/transforms/rewrite_regions.h"
#include "shardy/dialect/sdy/ir/register.h"
#include "stablehlo/dialect/Register.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    llvm::errs()
        << "Usage: run_shardy <file.mlir> "
           "[--round-trip] [--stop-after=propagation|reshards|collectives] "
           "[--dump-dir=directory]\n";
    return 1;
  }
  try {
    bool roundTripEnabled = false;
    ShardyRunOptions options;
    for (int i = 2; i < argc; ++i) {
      std::string argument = argv[i];
      if (argument == "--round-trip")
        roundTripEnabled = true;
      else if (argument.compare(0, 11, "--dump-dir=") == 0) {
        options.dumpDirectory = argument.substr(11);
      } else if (argument.compare(0, 13, "--stop-after=") == 0) {
        std::string stage = argument.substr(13);
        if (stage == "propagation")
          options.stopAfter = ShardyStage::Propagation;
        else if (stage == "reshards")
          options.stopAfter = ShardyStage::ExplicitReshards;
        else if (stage == "collectives")
          options.stopAfter = ShardyStage::Collectives;
        else
          throw std::invalid_argument("unknown stop stage: " + stage);
      } else {
        throw std::invalid_argument("unknown option: " + argument);
      }
    }
    mlir::DialectRegistry registry;
    mlir::stablehlo::registerAllDialects(registry);
    mlir::sdy::registerAllDialects(registry);
    mlir::MLIRContext context(registry);
    auto original = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
    if (!original || mlir::failed(mlir::verify(*original))) return 1;

    mlir::OwningOpRef<mlir::ModuleOp> module(original->clone());
    if (roundTripEnabled && mlir::failed(rewriteUnconstrainedRegions(*module)))
      return 1;
    ShardyRunner runner;
    if (mlir::failed(runner.run(*module, options))) return 1;
    for (const auto& snapshot : runner.snapshots()) {
      unsigned reshards = 0;
      unsigned collectives = 0;
      for (const auto& [op, count] : snapshot.communicationOps) {
        if (op == "sdy.reshard")
          reshards += count;
        else
          collectives += count;
      }
      llvm::outs() << snapshot.name << ": reshards=" << reshards
                   << ", collectives=" << collectives;
      for (const auto& [op, count] : snapshot.communicationOps) {
        llvm::outs() << ", " << op << "=" << count;
      }
      llvm::outs() << "\n";
    }
    llvm::outs() << "Verified resulting module:\n"
                 << runner.snapshots().back().mlir;
  } catch (const std::exception& error) {
    llvm::errs() << error.what() << "\n";
    return 1;
  }
  return 0;
}

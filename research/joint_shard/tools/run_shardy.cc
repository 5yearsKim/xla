#include <charconv>
#include <stdexcept>
#include <string>
#include <vector>

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
        << "Usage: run_shardy <file.mlir> [--round-trip] [--candidates=1..4] "
           "[--stop-after=propagation|reshards|collectives] "
           "[--dump-dir=directory] "
        << tensorRewriteOptionHelp() << "\n";
    return 1;
  }
  try {
    bool roundTrip = false, showReport = false;
    unsigned candidates = 0;
    ShardyRunOptions shardyOptions;
    TensorRewriteOptions rewriteOptions;
    for (int i = 2; i < argc; ++i) {
      std::string argument = argv[i];
      if (argument == "--round-trip")
        roundTrip = true;
      else if (argument == "--rewrite-report")
        showReport = true;
      else if (argument.starts_with("--candidates=")) {
        auto text = std::string_view(argument).substr(13);
        auto [end, error] =
            std::from_chars(text.data(), text.data() + text.size(), candidates);
        if (error != std::errc{} || end != text.data() + text.size() ||
            candidates < 1 || candidates > 4)
          throw std::invalid_argument(
              "candidates must be between 1 and 4, including the baseline");
      } else if (argument.starts_with("--dump-dir="))
        shardyOptions.dumpDirectory = argument.substr(11);
      else if (argument.starts_with("--stop-after=")) {
        auto stage = argument.substr(13);
        if (stage == "propagation")
          shardyOptions.stopAfter = ShardyStage::Propagation;
        else if (stage == "reshards")
          shardyOptions.stopAfter = ShardyStage::ExplicitReshards;
        else if (stage == "collectives")
          shardyOptions.stopAfter = ShardyStage::Collectives;
        else
          throw std::invalid_argument("unknown stop stage: " + stage);
      } else if (!parseTensorRewriteOption(argument, rewriteOptions))
        throw std::invalid_argument("unknown option: " + argument);
    }
    if (!candidates) candidates = roundTrip ? 2 : 1;
    if (candidates > 1 && shardyOptions.stopAfter != ShardyStage::Collectives)
      throw std::invalid_argument(
          "candidate comparison requires --stop-after=collectives");
    if (candidates > 1) {
      if (rewriteOptions.rules_file.empty())
        parseSemanticRules(defaultSemanticRules(),
                           rewriteOptions.numerical_policy,
                           rewriteOptions.semantic);
      else
        loadSemanticRulesFile(rewriteOptions.rules_file,
                              rewriteOptions.numerical_policy,
                              rewriteOptions.semantic);
    }
    mlir::DialectRegistry registry;
    mlir::stablehlo::registerAllDialects(registry);
    mlir::sdy::registerAllDialects(registry);
    mlir::MLIRContext context(registry);
    auto original = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
    if (!original || mlir::failed(mlir::verify(*original))) return 1;
    const char* names[] = {"original", "compute", "depth", "memory"};
    const ExtractionProfile profiles[] = {
        ExtractionProfile::Compute, ExtractionProfile::Compute,
        ExtractionProfile::Depth, ExtractionProfile::Memory};
    ModuleCost bestCost;
    std::vector<ShardySnapshot> bestSnapshots;
    unsigned best = 0;
    for (unsigned index = 0; index < candidates; ++index) {
      mlir::OwningOpRef<mlir::ModuleOp> module(original->clone());
      TensorRewriteReport report;
      auto candidateOptions = rewriteOptions;
      if (index) {
        // Respect an explicitly selected extraction profile for the first
        // candidate.
        candidateOptions.extraction =
            index == 1 ? rewriteOptions.extraction : profiles[index];
        try {
          if (mlir::failed(rewriteUnconstrainedRegions(
                  *module, candidateOptions, &report)))
            continue;
        } catch (const std::exception& error) {
          llvm::errs() << "rejected candidate " << names[index] << ": "
                       << error.what() << "\n";
          continue;
        }
        if (showReport)
          llvm::errs() << "candidate " << names[index] << "\n" << report.str();
      }
      ShardyRunner runner;
      auto options = shardyOptions;
      if (!options.dumpDirectory.empty() && candidates > 1)
        options.dumpDirectory += "/" + std::string(names[index]);
      try {
        if (mlir::failed(runner.run(*module, options))) {
          if (!index) return 1;
          continue;
        }
      } catch (const std::exception& error) {
        if (!index) throw;
        llvm::errs() << "rejected candidate " << names[index] << ": "
                     << error.what() << "\n";
        continue;
      }
      auto cost = runner.snapshots().back().cost;
      if (index == 0 || betterModuleCost(cost, bestCost)) {
        best = index;
        bestCost = cost;
        bestSnapshots = runner.snapshots();
      }
      llvm::errs() << "candidate " << names[index]
                   << " payload_bytes=" << cost.communication_payload_bytes
                   << " compute_work=" << cost.compute_work
                   << " unknown_costs=" << cost.unknown_costs << "\n";
    }
    if (bestSnapshots.empty()) return 1;
    llvm::errs() << "selected=" << names[best]
                 << " (logical payload estimate, then compute work; ties keep "
                    "original)\n";
    llvm::outs() << bestSnapshots.back().mlir;
  } catch (const std::exception& error) {
    llvm::errs() << error.what() << "\n";
    return 1;
  }
  return 0;
}

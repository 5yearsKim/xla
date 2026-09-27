#include <exception>
#include <stdexcept>
#include <string>

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "cxxopts.hpp"
#include "research/joint_shard/reporting/artifact_writer.h"
#include "research/joint_shard/sharding/shardy_runner.h"
#include "shardy/dialect/sdy/ir/register.h"
#include "stablehlo/dialect/Register.h"

namespace joint_shard {

// Raw pipeline inspection. Rewrite/layout search belongs to summarize_regions.
}  // namespace joint_shard

int main(int argc, char** argv) {
  using namespace joint_shard;
  try {
    cxxopts::Options cli("run_shardy",
                         "Run the raw Shardy pipeline on an MLIR file.");
    cli.positional_help("<file.mlir>");
    cli.custom_help("<file.mlir> [options]");
    cli.add_options()("input", "Input MLIR file",
                      cxxopts::value<std::string>())("h,help",
                                                     "Show this help")(
        "stop-after", "Stop after propagation, reshards, or collectives",
        cxxopts::value<std::string>())(
        "dump-dir", "Write pipeline snapshots to this directory",
        cxxopts::value<std::string>());
    cli.parse_positional({"input"});
    const cxxopts::ParseResult parsed = cli.parse(argc, argv);
    if (parsed.count("help")) {
      llvm::outs() << cli.help() << '\n';
      return 0;
    }
    if (!parsed.count("input") || !parsed.unmatched().empty()) {
      llvm::errs() << cli.help() << '\n';
      return 1;
    }

    ShardyRunOptions options;
    options.collect_statistics = true;
    std::string dump_directory;
    for (const cxxopts::KeyValue& argument : parsed.arguments()) {
      if (argument.key() == "input") continue;
      if (argument.key() == "dump-dir") {
        dump_directory = argument.value();
      } else if (argument.key() == "stop-after") {
        const std::string& stage = argument.value();
        if (stage == "propagation")
          options.stop_after = ShardyStage::Propagation;
        else if (stage == "reshards")
          options.stop_after = ShardyStage::ExplicitReshards;
        else if (stage == "collectives")
          options.stop_after = ShardyStage::Collectives;
        else
          throw std::invalid_argument("unknown stop stage: " + stage);
      }
    }

    mlir::DialectRegistry registry;
    mlir::stablehlo::registerAllDialects(registry);
    mlir::sdy::registerAllDialects(registry);
    mlir::MLIRContext context(registry);
    auto module = mlir::parseSourceFile<mlir::ModuleOp>(
        parsed["input"].as<std::string>(), &context);
    if (!module || mlir::failed(mlir::verify(*module))) return 1;
    ArtifactWriter artifacts(dump_directory);
    if (!dump_directory.empty()) {
      options.capture = SnapshotCapture::AllStages;
      options.on_snapshot = [&](const ShardySnapshot& snapshot) {
        artifacts.writeSnapshot(snapshot);
      };
    }
    ShardyRunner runner;
    if (mlir::failed(runner.run(*module, options))) return 1;
    const auto& snapshot = runner.snapshots().back();
    llvm::errs() << "logical_payload_bytes="
                 << snapshot.statistics->communication_payload_bytes
                 << " global_compute_work=" << snapshot.statistics->compute_work
                 << " unknown_costs=" << snapshot.statistics->unknown_costs
                 << "\n";
    llvm::outs() << snapshot.mlir;
  } catch (const std::exception& error) {
    llvm::errs() << error.what() << "\n";
    return 1;
  }
  return 0;
}

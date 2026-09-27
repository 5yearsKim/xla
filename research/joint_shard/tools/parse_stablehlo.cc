#include <exception>
#include <stdexcept>
#include <string>

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "cxxopts.hpp"
#include "research/joint_shard/tools/rewrite_cli_options.h"
#include "research/joint_shard/transforms/rewrite_regions.h"
#include "shardy/dialect/sdy/ir/register.h"
#include "stablehlo/dialect/Register.h"

int main(int argc, char** argv) {
  try {
    cxxopts::Options cli(
        "parse_stablehlo",
        "Rewrite supported StableHLO regions in an MLIR file.");
    cli.positional_help("<file.mlir>");
    cli.custom_help("<file.mlir> [options]");
    cli.add_options()("input", "Input MLIR file",
                      cxxopts::value<std::string>())(
        "h,help", "Show this help")("rewrite-report",
                                    "Print rewrite diagnostics to stderr");
    cli.parse_positional({"input"});
    addRewriteCliOptions(cli);
    const cxxopts::ParseResult parsed = cli.parse(argc, argv);
    if (parsed.count("help")) {
      llvm::outs() << cli.help() << '\n';
      return 0;
    }
    if (!parsed.count("input") || !parsed.unmatched().empty()) {
      llvm::errs() << cli.help() << '\n';
      return 1;
    }

    TensorRewriteOptions options;
    bool show_report = false;
    for (const cxxopts::KeyValue& argument : parsed.arguments()) {
      if (argument.key() == "input") continue;
      if (argument.key() == "rewrite-report") {
        show_report = true;
      } else if (!applyRewriteCliOption(argument.key(), argument.value(),
                                        options)) {
        throw std::invalid_argument("unknown option: --" + argument.key());
      }
    }

    mlir::DialectRegistry registry;
    mlir::stablehlo::registerAllDialects(registry);
    mlir::sdy::registerAllDialects(registry);
    mlir::MLIRContext context(registry);
    auto module = mlir::parseSourceFile<mlir::ModuleOp>(
        parsed["input"].as<std::string>(), &context);
    if (!module || mlir::failed(mlir::verify(*module))) return 1;
    TensorRewriteReport report;
    if (mlir::failed(rewriteUnconstrainedRegions(*module, options, &report)))
      return 1;
    if (show_report) llvm::errs() << report.str();
    module->print(llvm::outs());
    llvm::outs() << "\n";
  } catch (const std::exception& error) {
    llvm::errs() << error.what() << "\n";
    return 1;
  }
  return 0;
}

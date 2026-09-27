#include <charconv>
#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>

#include "llvm/Support/raw_ostream.h"
#include "mlir/Parser/Parser.h"
#include "cxxopts.hpp"
#include "research/joint_shard/search/region_optimizer.h"
#include "research/joint_shard/tools/rewrite_cli_options.h"
#include "shardy/dialect/sdy/ir/register.h"
#include "stablehlo/dialect/Register.h"

namespace {
size_t positive(std::string_view text) {
  size_t value = 0;
  auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size() || !value)
    throw std::invalid_argument("expected positive integer: " +
                                std::string(text));
  return value;
}
std::optional<int64_t> dimension(std::string_view text) {
  if (text == "none") return {};
  int64_t value = 0;
  auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size())
    throw std::invalid_argument("expected dimension index or none");
  return value;
}
double number(std::string_view text) {
  size_t consumed;
  double value = std::stod(std::string(text), &consumed);
  if (consumed != text.size() || !std::isfinite(value))
    throw std::invalid_argument("expected finite number");
  return value;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    cxxopts::Options cli("summarize_regions",
                         "Search bounded sharding plans for MLIR regions.");
    cli.positional_help("<file.mlir>");
    cli.custom_help("<file.mlir> [options]");
    cli.add_options()("input", "Input MLIR file",
                      cxxopts::value<std::string>())(
        "h,help", "Show this help")("mesh", "Mesh name to use",
                                    cxxopts::value<std::string>())(
        "max-region-ops", "Maximum operations per region",
        cxxopts::value<std::string>())("max-candidates",
                                       "Maximum candidates per region",
                                       cxxopts::value<std::string>())(
        "max-boundary-states", "Maximum boundary states",
        cxxopts::value<std::string>())("compose-regions",
                                       "Compose adjacent region indices A,B",
                                       cxxopts::value<std::string>())(
        "compose-pruned", "Include candidates removed by pruning")(
        "max-pair-evaluations", "Maximum pair evaluations",
        cxxopts::value<std::string>())("data-axis",
                                       "Mesh axis for data parallelism",
                                       cxxopts::value<std::string>())(
        "model-axis", "Mesh axis for model parallelism",
        cxxopts::value<std::string>())(
        "data-dim", "Default tensor dimension for data axis, or none",
        cxxopts::value<std::string>())(
        "model-dim", "Default tensor dimension for model axis, or none",
        cxxopts::value<std::string>())(
        "input-policy", "Input port policy PORT:DATA_DIM:MODEL_DIM",
        cxxopts::value<std::string>())(
        "output-policy", "Output port policy PORT:DATA_DIM:MODEL_DIM",
        cxxopts::value<std::string>())("rule-blocker",
                                       "Rule blocker operation name, or none",
                                       cxxopts::value<std::string>())(
        "compute-work-per-us", "Compute work per microsecond",
        cxxopts::value<std::string>())("bandwidth-bytes-per-us",
                                       "Bandwidth in bytes per microsecond",
                                       cxxopts::value<std::string>())(
        "collective-latency-us", "Collective latency in microseconds",
        cxxopts::value<std::string>())(
        "dump-dir", "Write intermediate IR to this directory",
        cxxopts::value<std::string>());
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

    RegionOptimizerOptions options;
    for (const cxxopts::KeyValue& argument : parsed.arguments()) {
      const std::string& name = argument.key();
      const std::string& value = argument.value();
      if (name == "input") continue;
      if (applyRewriteCliOption(name, value, options.rewriting)) continue;
      if (name == "compose-pruned")
        options.compose_pruned = true;
      else if (name == "max-pair-evaluations")
        options.max_pair_evaluations = positive(value);
      else if (name == "compose-regions") {
        auto comma = value.find(',');
        if (comma == std::string_view::npos)
          throw std::invalid_argument("expected A,B region indices");
        auto index = [](std::string_view text) {
          size_t result = 0;
          auto [end, error] =
              std::from_chars(text.data(), text.data() + text.size(), result);
          if (error != std::errc{} || end != text.data() + text.size())
            throw std::invalid_argument("invalid region index");
          return result;
        };
        options.compose_regions = {
            {index(value.substr(0, comma)), index(value.substr(comma + 1))}};
      } else if (name == "max-region-ops")
        options.regionizer.max_region_ops = positive(value);
      else if (name == "max-candidates")
        options.max_candidates = positive(value);
      else if (name == "max-boundary-states")
        options.max_boundary_states = positive(value);
      else if (name == "mesh")
        options.mesh_name = value;
      else if (name == "dump-dir")
        options.dump_directory = value;
      else if (name == "data-axis")
        options.layouts.data_axis = value;
      else if (name == "model-axis")
        options.layouts.model_axis = value;
      else if (name == "data-dim")
        options.layouts.defaults.data_dimension = dimension(value);
      else if (name == "model-dim")
        options.layouts.defaults.model_dimension = dimension(value);
      else if (name == "rule-blocker") {
        if (value == "none")
          options.regionizer.rule_blockers.clear();
        else if (!value.empty())
          options.regionizer.rule_blockers.emplace_back(value);
        else
          throw std::invalid_argument("empty rule blocker");
      } else if (name == "input-policy" || name == "output-policy") {
        auto first = value.find(':'), last = value.rfind(':');
        if (first == std::string_view::npos || first == last)
          throw std::invalid_argument(
              "port policy must be PORT:DATA_DIM:MODEL_DIM");
        size_t port = 0;
        auto [end, error] =
            std::from_chars(value.data(), value.data() + first, port);
        if (error != std::errc{} || end != value.data() + first)
          throw std::invalid_argument("invalid port index");
        DimensionPolicy policy{
            dimension(value.substr(first + 1, last - first - 1)),
            dimension(value.substr(last + 1))};
        (name == "input-policy" ? options.layouts.inputs
                                : options.layouts.outputs)[port] = policy;
      } else if (name == "compute-work-per-us")
        options.cost.compute_work_per_us = number(value);
      else if (name == "bandwidth-bytes-per-us")
        options.cost.bandwidth_bytes_per_us = number(value);
      else if (name == "collective-latency-us")
        options.cost.collective_latency_us = number(value);
      else
        throw std::invalid_argument("unknown option: --" + name);
    }
    mlir::DialectRegistry registry;
    mlir::stablehlo::registerAllDialects(registry);
    mlir::sdy::registerAllDialects(registry);
    mlir::MLIRContext context(registry);
    auto module = mlir::parseSourceFile<mlir::ModuleOp>(
        parsed["input"].as<std::string>(), &context);
    if (!module) return 1;
    auto report = summarizeRegions(*module, options);
    llvm::outs() << report.str();
  } catch (const std::exception& error) {
    llvm::errs() << error.what() << '\n';
    return 1;
  }
  return 0;
}

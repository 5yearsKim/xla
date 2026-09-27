#include <charconv>
#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>

#include "llvm/Support/raw_ostream.h"
#include "mlir/Parser/Parser.h"
#include "research/joint_shard/search/region_optimizer.h"
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
  if (argc < 2) {
    llvm::errs()
        << "Usage: summarize_regions <file.mlir> [--mesh=name] "
           "[--max-region-ops=128] [--max-candidates=32] "
           "[--max-boundary-states=256] "
           "[--data-axis=data] [--model-axis=model] [--data-dim=0|none] "
           "[--model-dim=-1|none] [--input-policy=PORT:DATA_DIM:MODEL_DIM] "
           "[--output-policy=PORT:DATA_DIM:MODEL_DIM] [--rule-blocker=op|none] "
           "[--compute-work-per-us=1000000] [--bandwidth-bytes-per-us=50000] "
           "[--collective-latency-us=5] [--dump-dir=path] "
        << tensorRewriteOptionHelp()
        << "\nDefault numerical policy: relaxed.\n";
    return 1;
  }
  try {
    RegionOptimizerOptions options;
    for (int i = 2; i < argc; ++i) {
      std::string_view argument(argv[i]);
      if (parseTensorRewriteOption(argument, options.rewriting)) continue;
      auto at = argument.find('=');
      auto name = argument.substr(0, at);
      auto value = at == std::string_view::npos ? std::string_view{}
                                                : argument.substr(at + 1);
      if (name == "--max-region-ops")
        options.regionizer.max_region_ops = positive(value);
      else if (name == "--max-candidates")
        options.max_candidates = positive(value);
      else if (name == "--max-boundary-states")
        options.max_boundary_states = positive(value);
      else if (name == "--mesh")
        options.mesh_name = value;
      else if (name == "--dump-dir")
        options.dump_directory = value;
      else if (name == "--data-axis")
        options.layouts.data_axis = value;
      else if (name == "--model-axis")
        options.layouts.model_axis = value;
      else if (name == "--data-dim")
        options.layouts.defaults.data_dimension = dimension(value);
      else if (name == "--model-dim")
        options.layouts.defaults.model_dimension = dimension(value);
      else if (name == "--rule-blocker") {
        if (value == "none")
          options.regionizer.rule_blockers.clear();
        else if (!value.empty())
          options.regionizer.rule_blockers.emplace_back(value);
        else
          throw std::invalid_argument("empty rule blocker");
      } else if (name == "--input-policy" || name == "--output-policy") {
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
        (name == "--input-policy" ? options.layouts.inputs
                                  : options.layouts.outputs)[port] = policy;
      } else if (name == "--compute-work-per-us")
        options.cost.compute_work_per_us = number(value);
      else if (name == "--bandwidth-bytes-per-us")
        options.cost.bandwidth_bytes_per_us = number(value);
      else if (name == "--collective-latency-us")
        options.cost.collective_latency_us = number(value);
      else
        throw std::invalid_argument("unknown option: " + std::string(argument));
    }
    mlir::DialectRegistry registry;
    mlir::stablehlo::registerAllDialects(registry);
    mlir::sdy::registerAllDialects(registry);
    mlir::MLIRContext context(registry);
    auto module = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
    if (!module) return 1;
    auto report = summarizeRegions(*module, options);
    llvm::outs() << report.str();
  } catch (const std::exception& error) {
    llvm::errs() << error.what() << '\n';
    return 1;
  }
  return 0;
}

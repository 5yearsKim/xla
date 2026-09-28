// Bounded four-arm experiment for one fully supported, single-output workload.
// All arms retain one external contract; input layout changes pay adapters.
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <stdexcept>

#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "cxxopts.hpp"
#include "research/joint_shard/bridge/stablehlo_importer.h"
#include "research/joint_shard/export/selected_export.h"
#include "research/joint_shard/sharding/module_statistics.h"
#include "research/joint_shard/sharding/plan_materializer.h"
#include "research/joint_shard/sharding/region_evaluator.h"
#include "research/joint_shard/tools/rewrite_cli_options.h"
#include "research/joint_shard/transforms/region_candidates.h"
#include "shardy/dialect/sdy/ir/register.h"
#include "stablehlo/dialect/Register.h"

namespace joint_shard {
namespace {
void write(const std::filesystem::path& path, const std::string& text) {
  std::ofstream file(path);
  file << text;
  if (!file) throw std::runtime_error("failed to write " + path.string());
}
std::string print(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream out(text);
  module.print(out);
  return text;
}
llvm::json::Object costJson(const Cost& cost) {
  return llvm::json::Object{{"compute_us", cost.compute},
                            {"communication_us", cost.communication},
                            {"total_us", cost.total()},
                            {"unknown", cost.unknown}};
}
llvm::json::Array layoutsJson(const std::vector<TensorSharding>& layouts) {
  llvm::json::Array result;
  for (const auto& layout : layouts) result.push_back(layout.str());
  return result;
}
struct Winner {
  bool feasible = false;
  size_t candidate = 0, layout = 0;
  Cost cost, adapters_cost;
  std::string core;
  std::vector<ReshardPlan> adapters;
};
// Cost profiles can omit compute-expensive, communication-saving alternatives.
// Keep one direct application per target rule as an explicit experiment
// candidate. This uses the real checked pattern appliers on the original graph;
// no optimized Python workload or hand-built replacement is substituted.
void appendRuleWitnesses(const Candidate& original,
                         const TensorRewriteOptions& options,
                         std::vector<Candidate>& candidates) {
  auto semantic = options.semantic;
  semantic.report.reset();  // Saturation counters remain independent.
  for (const auto& rule :
       buildSemanticRules(options.numerical_policy, semantic)) {
    if (rule.name != "project-before-sequence-sum" &&
        rule.name != "reassociate-batched-shared-weight-right" &&
        !rule.name.starts_with("kernel-attention-summaries") &&
        !rule.name.starts_with("centered-square-to-raw-moments"))
      continue;
    TensorEGraph graph;
    std::vector<eggc::Id> ids;
    for (auto node : original.expression.nodes) {
      for (auto& child : node.operands) child = ids.at(child);
      ids.push_back(graph.add(std::move(node)));
    }
    graph.rebuild();
    std::optional<eggc::Application<TensorNode, TensorAnalysis>> match;
    rule.custom_search(graph,
                       [&](auto application) {
                         if (!application.apply) return true;
                         match = std::move(application);
                         return false;
                       },
                       {});
    if (!match) continue;
    auto replacement = match->apply(graph);
    if (!replacement) continue;
    std::vector<eggc::Id> rewritten;
    for (size_t i = 0; i < original.expression.nodes.size(); ++i) {
      if (graph.find(ids[i]) == graph.find(match->target)) {
        rewritten.push_back(*replacement);
      } else {
        auto node = original.expression.nodes[i];
        for (auto& child : node.operands) child = rewritten.at(child);
        rewritten.push_back(graph.add(std::move(node)));
      }
    }
    graph.rebuild();
    std::vector<eggc::Id> roots;
    for (auto root : original.output_roots) roots.push_back(rewritten.at(root));
    TensorExtractionReport ignored;
    auto candidate = mergeExtractedRoots(
        extractTensorRoots(graph, roots, ExtractionProfile::Compute,
                           TensorExtractorMode::Tree, options.dag, ignored),
        "witness-" + rule.name);
    bool duplicate = false;
    for (const auto& previous : candidates)
      duplicate |= previous.expression.nodes == candidate.expression.nodes &&
                   previous.output_roots == candidate.output_roots;
    if (!duplicate) {
      candidate.id = candidates.size();
      candidates.push_back(std::move(candidate));
    }
  }
}
}  // namespace
}  // namespace joint_shard

int main(int argc, char** argv) {
  using namespace joint_shard;
  try {
    cxxopts::Options cli("workload_ablation",
                         "Four arms with charged input conversions and a fixed "
                         "output contract.");
    cli.add_options()("input", "Supported single-output MLIR workload",
                      cxxopts::value<std::string>())(
        "output-dir", "Artifact directory", cxxopts::value<std::string>())(
        "max-layouts", "Input layout assignment cap",
        cxxopts::value<size_t>()->default_value("256"))(
        "compute-work-per-us", "Illustrative compute rate",
        cxxopts::value<double>()->default_value("1000000"))(
        "bandwidth-bytes-per-us", "Illustrative bandwidth",
        cxxopts::value<double>()->default_value("50000"))(
        "collective-latency-us", "Illustrative latency",
        cxxopts::value<double>()->default_value("5"))("h,help", "Show help");
    addRewriteCliOptions(cli);
    cli.parse_positional({"input"});
    auto parsed = cli.parse(argc, argv);
    if (parsed.count("help")) {
      llvm::outs() << cli.help() << '\n';
      return 0;
    }
    if (!parsed.count("input") || !parsed.count("output-dir") ||
        !parsed.unmatched().empty())
      throw std::invalid_argument("input and --output-dir are required");
    TensorRewriteOptions rewriting;
    rewriting.semantic.report =
        std::make_shared<std::map<std::string, SemanticRuleStats>>();
    for (const auto& arg : parsed.arguments())
      if (arg.key() != "input" && arg.key() != "output-dir" &&
          arg.key() != "max-layouts" && arg.key() != "compute-work-per-us" &&
          arg.key() != "bandwidth-bytes-per-us" &&
          arg.key() != "collective-latency-us")
        if (!applyRewriteCliOption(arg.key(), arg.value(), rewriting))
          throw std::invalid_argument("unknown rewrite option");
    const size_t cap = parsed["max-layouts"].as<size_t>();
    if (!cap) throw std::invalid_argument("max-layouts must be positive");
    CostModelOptions rates{parsed["compute-work-per-us"].as<double>(),
                           parsed["bandwidth-bytes-per-us"].as<double>(),
                           parsed["collective-latency-us"].as<double>()};
    CostModel model(rates);
    mlir::DialectRegistry registry;
    mlir::stablehlo::registerAllDialects(registry);
    mlir::sdy::registerAllDialects(registry);
    mlir::MLIRContext context(registry);
    auto module = mlir::parseSourceFile<mlir::ModuleOp>(
        parsed["input"].as<std::string>(), &context);
    if (!module || mlir::failed(mlir::verify(*module)))
      throw std::invalid_argument("invalid input MLIR");
    auto main = module->lookupSymbol<mlir::func::FuncOp>("main");
    if (!main || main.isExternal() || !main.getBody().hasOneBlock() ||
        main.getNumResults() != 1)
      throw std::invalid_argument(
          "experiment requires a single-block, single-output @main");
    auto mesh = selectMesh(*module);
    if (mesh.mesh.getAxes().size() != 1 || !mesh.mesh.hasAxis("model"))
      throw std::invalid_argument(
          "experiment supports a single model mesh axis");
    auto contract = functionLayoutContract(main, mesh);
    std::vector<mlir::Operation*> operations;
    for (auto& op : main.getBody().front().without_terminator()) {
      if (!canImportTensorOperation(&op))
        throw std::invalid_argument("unsupported workload operation: " +
                                    op.getName().getStringRef().str() + ": " +
                                    tensorImportRejection(&op));
      operations.push_back(&op);
    }
    if (operations.empty()) throw std::invalid_argument("empty workload");
    auto region = describeRegion(operations);
    if (region.outputs.size() != 1 ||
        region.inputs.size() != main.getNumArguments())
      throw std::invalid_argument(
          "experiment requires all inputs used and one produced result");
    BoundaryState external;
    external.outputs = contract.outputs;
    llvm::json::Array argument_order;
    std::vector<mlir::Type> input_types;
    for (auto value : region.inputs) {
      auto argument = llvm::dyn_cast<mlir::BlockArgument>(value);
      if (!argument || argument.getOwner() != &main.getBody().front())
        throw std::invalid_argument(
            "workload input is not a function argument");
      argument_order.push_back(argument.getArgNumber());
      external.inputs.push_back(contract.inputs.at(argument.getArgNumber()));
      input_types.push_back(value.getType());
    }
    // Saturation imports arithmetic only; output layout is imposed at
    // evaluation.
    auto saturated =
        saturateRegion(region, compileTensorRules(rewriting), rewriting);
    CandidateExtractionReport extraction;
    auto candidates = extractCandidates(saturated, rewriting, 32, &extraction);
    const auto profile_candidates = candidates.size();
    appendRuleWitnesses(saturated.original, rewriting, candidates);
    std::map<void*, TensorSharding> fixed_outputs{
        {region.outputs[0].getAsOpaquePointer(), external.outputs[0]}};
    std::vector<mlir::OwningOpRef<mlir::ModuleOp>> prepared;
    for (const auto& candidate : candidates)
      prepared.push_back(
          prepareCandidateModule(region, candidate, mesh, fixed_outputs));
    // Search replicated or one full model-axis partition on any divisible input
    // dimension. No output factor: every arm has the same exact output
    // contract.
    std::vector<std::vector<TensorSharding>> choices;
    LayoutPolicy policy;
    policy.data_axis.clear();
    for (auto value : region.inputs) {
      auto type = llvm::cast<mlir::RankedTensorType>(value.getType());
      if (!type.hasStaticShape())
        throw std::invalid_argument("static inputs required");
      std::vector<TensorSharding> port{replicatedSharding(type, mesh)};
      for (int64_t dim = 0; dim < type.getRank(); ++dim)
        for (auto layout :
             tensorLayoutChoices(type, mesh, policy, {std::nullopt, dim}))
          if (std::find(port.begin(), port.end(), layout) == port.end())
            port.push_back(layout);
      choices.push_back(std::move(port));
    }
    std::vector<std::vector<TensorSharding>> states{external.inputs};
    bool truncated = false;
    std::vector<TensorSharding> assignment;
    std::function<bool(size_t)> enumerate = [&](size_t port) {
      if (port == choices.size()) {
        if (assignment == external.inputs) return true;
        if (states.size() == cap) {
          truncated = true;
          return false;
        }
        states.push_back(assignment);
        return true;
      }
      for (auto layout : choices[port]) {
        assignment.push_back(layout);
        bool keep_going = enumerate(port + 1);
        assignment.pop_back();
        if (!keep_going) return false;
      }
      return true;
    };
    enumerate(0);
    RegionEvaluator evaluator(mesh, model);
    ReshardCostOracle oracle(mesh, model);
    std::array<Winner, 4> winners;
    size_t evaluations = 0;
    llvm::json::Array failures;
    for (size_t layout = 0; layout < states.size(); ++layout) {
      std::vector<ReshardPlan> adapters;
      Cost adapters_cost;
      bool feasible = true;
      for (size_t i = 0; i < states[layout].size(); ++i) {
        auto adapter =
            oracle.plan(external.inputs[i], states[layout][i], input_types[i]);
        feasible &= adapter.feasible;
        adapters_cost += adapter.cost;
        adapters.push_back(std::move(adapter));
      }
      if (!feasible) {
        failures.push_back("infeasible input conversion");
        continue;
      }
      for (size_t candidate = 0; candidate < prepared.size(); ++candidate) {
        ++evaluations;
        auto evaluated =
            evaluator.evaluate(*prepared[candidate], states[layout]);
        if (!evaluated.feasible || !evaluated.cost.known()) {
          failures.push_back(evaluated.failure.empty() ? "unknown cost"
                                                       : evaluated.failure);
          continue;
        }
        auto cost = evaluated.cost;
        cost += adapters_cost;
        const std::array<bool, 4> eligible{layout == 0 && candidate == 0,
                                           layout == 0, candidate == 0, true};
        for (size_t arm = 0; arm < winners.size(); ++arm)
          if (eligible[arm] && (!winners[arm].feasible ||
                                cost.total() < winners[arm].cost.total()))
            winners[arm] = {true,    candidate,     layout,
                            cost,    adapters_cost, evaluated.lowered_mlir,
                            adapters};
      }
    }
    const std::filesystem::path directory(
        parsed["output-dir"].as<std::string>());
    std::filesystem::create_directories(directory);
    llvm::json::Object arms;
    const std::array<const char*, 4> names{"baseline", "rewrite_only",
                                           "sharding_only", "combined"};
    for (size_t arm = 0; arm < winners.size(); ++arm) {
      const auto& selected = winners[arm];
      if (!selected.feasible)
        throw std::runtime_error("no feasible " + std::string(names[arm]) +
                                 " arm");
      PlanMaterializer materializer(mesh, input_types,
                                    {region.outputs[0].getType()}, external);
      std::vector<mlir::Value> inputs(materializer.arguments().begin(),
                                      materializer.arguments().end());
      for (size_t i = 0; i < inputs.size(); ++i)
        if (!selected.adapters[i].lowered_mlir.empty())
          inputs[i] = materializer
                          .inlineArtifact(selected.adapters[i].lowered_mlir,
                                          {inputs[i]})
                          .at(0);
      auto outputs = materializer.inlineArtifact(selected.core, inputs);
      auto text = materializer.finish(outputs, selected.cost, model);
      auto selected_module =
          mlir::parseSourceString<mlir::ModuleOp>(text, &context);
      auto stats = collectModuleStatistics(*selected_module);
      uint64_t collective_count = 0;
      llvm::json::Object collectives;
      for (const auto& [name, count] : stats.communication_ops) {
        collectives[name] = count;
        if (name != "sdy.all_slice") collective_count += count;
      }
      const double bytes =
          std::max(0.0, (selected.cost.communication -
                         collective_count * rates.collective_latency_us) *
                            rates.bandwidth_bytes_per_us);
      auto exported = exportSelectedProgram(*selected_module);
      if (!exported.ok())
        throw std::runtime_error(exported.status().ToString());
      write(directory / (std::string(names[arm]) + ".selected.mlir"), text);
      write(directory / (std::string(names[arm]) + ".xla_input.mlir"),
            print(*exported->module));
      write(directory / (std::string(names[arm]) + ".candidate.mlir"),
            print(*prepared[selected.candidate]));
      arms[names[arm]] = llvm::json::Object{
          {"candidate_id", selected.candidate},
          {"candidate", candidates[selected.candidate].name},
          {"internal_inputs", layoutsJson(states[selected.layout])},
          {"cost", costJson(selected.cost)},
          {"input_adapters_cost", costJson(selected.adapters_cost)},
          {"collectives", std::move(collectives)},
          {"modeled_bytes_per_device", bytes},
          {"global_logical_payload_bytes", stats.communication_payload_bytes}};
    }
    llvm::json::Object rules;
    for (const auto& [name, stats] : *rewriting.semantic.report)
      rules[name] = llvm::json::Object{{"matches", stats.structural_matches},
                                       {"applied", stats.applied}};
    bool extraction_limited = false;
    for (const auto& profile : extraction.profiles)
      extraction_limited |=
          profile.stop && *profile.stop != eggc::DagStopReason::Exhausted;
    llvm::json::Array candidate_names;
    for (const auto& candidate : candidates)
      candidate_names.push_back(candidate.name);
    llvm::json::Object result{
        {"arms", std::move(arms)},
        {"argument_order", std::move(argument_order)},
        {"external_inputs", layoutsJson(external.inputs)},
        {"external_outputs", layoutsJson(external.outputs)},
        {"candidate_count", candidates.size()},
        {"profile_candidate_count", profile_candidates},
        {"candidate_names", std::move(candidate_names)},
        {"layout_count", states.size()},
        {"layout_truncated", truncated},
        {"evaluations", evaluations},
        {"failures", std::move(failures)},
        {"rules", std::move(rules)},
        {"saturation",
         llvm::json::Object{
             {"stop",
              std::array<const char*, 8>{"saturated", "iteration-limit",
                                         "node-limit", "time-limit",
                                         "match-limit", "search-limit",
                                         "user-requested", "memory-limit"}
                  .at(static_cast<size_t>(saturated.report.reason))},
             {"nodes", saturated.report.nodes},
             {"iterations", saturated.report.iterations}}},
        {"extraction_profiles_skipped", extraction.profiles_skipped_at_cap},
        {"extraction_limited", extraction_limited},
        {"cost_parameters",
         llvm::json::Object{
             {"compute_work_per_us", rates.compute_work_per_us},
             {"bandwidth_bytes_per_us", rates.bandwidth_bytes_per_us},
             {"collective_latency_us", rates.collective_latency_us}}}};
    std::string json;
    llvm::raw_string_ostream out(json);
    out << llvm::formatv("{0:2}", llvm::json::Value(std::move(result))) << '\n';
    write(directory / "result.json", json);
    llvm::outs() << json;
  } catch (const std::exception& error) {
    llvm::errs() << error.what() << '\n';
    return 1;
  }
  return 0;
}

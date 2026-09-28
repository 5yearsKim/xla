#include "research/joint_shard/search/region_optimizer.h"

#include <chrono>
#include <stdexcept>

#include "mlir/IR/Verifier.h"
#include "research/joint_shard/bridge/stablehlo_importer.h"
#include "research/joint_shard/sharding/region_evaluator.h"
#include "research/joint_shard/sharding/shardy_runner.h"
#include "research/joint_shard/transforms/region_candidates.h"

namespace joint_shard {

OptimizationReport summarizeRegions(mlir::ModuleOp module,
                                    const RegionOptimizerOptions& options,
                                    const OptimizationObserver& observer) {
  if (mlir::failed(mlir::verify(module)))
    throw std::invalid_argument("invalid input module");
  if (!options.max_candidates || !options.max_input_states ||
      !options.max_pair_evaluations || !options.max_chain_transitions ||
      !options.dag.max_live_values || !options.dag.max_states ||
      !options.dag.max_transitions)
    throw std::invalid_argument("search caps must be positive");
  if (unsigned(options.optimize_chain) + unsigned(options.optimize_dag) +
          unsigned(options.compose_regions.has_value()) >
      1)
    throw std::invalid_argument(
        "choose one of chain optimization, DAG optimization or pair "
        "composition");
  ValueIndex values(module);
  std::vector<Region> source_regions;
  OptimizationReport report;
  report.mesh = selectMesh(module, options.mesh_name);
  if (auto partitions =
          module->getAttrOfType<mlir::IntegerAttr>("mhlo.num_partitions"))
    if (partitions.getInt() != report.mesh.mesh.getTotalSize())
      throw std::invalid_argument(
          "mhlo.num_partitions disagrees with selected mesh");
  auto rules = compileTensorRules(options.rewriting);
  Regionizer regionizer(options.regionizer);
  CostModel model(options.cost);
  RegionEvaluator evaluator(report.mesh, model);
  ReshardCostOracle oracle(report.mesh, model);
  std::map<void*, TensorSharding> fixed_inputs, fixed_outputs;
  FunctionSearchStatistics* statistics = nullptr;
  if (options.optimize_chain || options.optimize_dag) {
    mlir::func::FuncOp main;
    size_t functions = 0;
    for (auto function : module.getOps<mlir::func::FuncOp>()) {
      ++functions;
      if (function.getSymName() == "main") main = function;
    }
    if (functions != 1 || !main)
      throw std::invalid_argument(
          "function optimization requires one function named @main");
    auto regions = regionizer.split(main);
    if (options.optimize_chain) {
      report.chain.emplace();
      report.chain->interface = buildChainInterface(main, regions, values);
      statistics = &*report.chain;
    } else {
      report.dag.emplace();
      report.dag->interface = buildDagInterface(main, regions, values);
      for (const auto& cut : report.dag->interface.live_after)
        if (cut.size() > options.dag.max_live_values)
          throw std::invalid_argument(
              "DAG live cut width " + std::to_string(cut.size()) +
              " exceeds max-live-values=" +
              std::to_string(options.dag.max_live_values));
      statistics = &*report.dag;
    }
    statistics->contract = functionLayoutContract(main, report.mesh);
    statistics->numerical_policy = options.rewriting.numerical_policy ==
                                           NumericalPolicy::PreserveEvaluation
                                       ? "strict"
                                       : "relaxed/explicit permissions";
    for (size_t i = 0; i < main.getNumArguments(); ++i)
      fixed_inputs.emplace(main.getArgument(i).getAsOpaquePointer(),
                           statistics->contract.inputs[i]);
    auto ret = llvm::cast<mlir::func::ReturnOp>(
        main.getBody().front().getTerminator());
    for (size_t i = 0; i < ret.getNumOperands(); ++i) {
      auto [it, inserted] =
          fixed_outputs.emplace(ret.getOperand(i).getAsOpaquePointer(),
                                statistics->contract.outputs[i]);
      if (!inserted && it->second != statistics->contract.outputs[i])
        throw std::invalid_argument(
            "repeated return value has inconsistent contracts");
      if (auto input =
              fixed_inputs.find(ret.getOperand(i).getAsOpaquePointer());
          input != fixed_inputs.end() &&
          input->second != statistics->contract.outputs[i])
        throw std::invalid_argument(
            "returned function argument has incompatible contracts");
    }
  }
  const auto evaluation_start = std::chrono::steady_clock::now();
  for (auto function : module.getOps<mlir::func::FuncOp>()) {
    if (function.isExternal()) continue;
    auto regions = regionizer.split(function);
    for (auto& op : function.getBody().front())
      if (!llvm::isa<mlir::func::ReturnOp>(op) &&
          !canImportTensorOperation(&op))
        ++report.preserved_operations[op.getName().getStringRef().str() + ": " +
                                      tensorImportRejection(&op)];
    for (auto& region : regions) {
      region.id = report.regions.size();
      auto saturated = saturateRegion(region, rules, options.rewriting);
      report.saturation.push_back(saturated.report);
      RegionSummary summary;
      source_regions.push_back(region);
      summary.interface = values.interface(region);
      summary.id = region.id;
      summary.operations = region.operations.size();
      summary.oversized = region.oversized;
      CandidateExtractionReport extraction;
      summary.candidates = extractCandidates(
          saturated, options.rewriting, options.max_candidates, &extraction);
      summary.candidate_profiles_skipped = extraction.profiles_skipped_at_cap;
      for (const auto& profile : extraction.profiles) {
        summary.extraction_states += profile.explored_states;
        summary.extraction_us +=
            std::chrono::duration<double, std::micro>(profile.search_time)
                .count();
        summary.extraction_limited |=
            profile.stop && *profile.stop != eggc::DagStopReason::Exhausted;
      }
      auto inputs =
          enumerateInputStates(region, report.mesh, options.layouts,
                               options.max_input_states, fixed_inputs);
      summary.input_search_truncated = inputs.truncated;
      summary.input_states_evaluated = inputs.states.size();
      if (inputs.states.empty())
        summary.failures
            ["no compatible input layouts (check static shapes, mesh, and "
             "constraints)"] = 1;
      std::vector<mlir::OwningOpRef<mlir::ModuleOp>> prepared;
      for (const auto& candidate : summary.candidates) {
        auto candidateModule = prepareCandidateModule(
            region, candidate, report.mesh, fixed_outputs);
        if (observer.candidate_prepared)
          observer.candidate_prepared(region.id, candidate.id,
                                      *candidateModule);
        prepared.push_back(std::move(candidateModule));
      }
      for (size_t b = 0; b < inputs.states.size(); ++b) {
        for (size_t c = 0; c < prepared.size(); ++c) {
          ShardyRunOptions runOptions;
          if (observer.snapshot) {
            runOptions.capture = SnapshotCapture::AllStages;
            runOptions.on_snapshot = [&](const ShardySnapshot& snapshot) {
              observer.snapshot(region.id, b, c, snapshot);
            };
          }
          auto evaluated =
              evaluator.evaluate(*prepared[c], inputs.states[b], runOptions);
          auto boundary = evaluated.boundary;
          summary.record(c, boundary, std::move(evaluated));
        }
      }
      summary.finalizePlans();
      if (statistics) {
        statistics->input_truncated |= summary.input_search_truncated;
        statistics->candidate_cap_reached |= summary.candidate_profiles_skipped;
        statistics->extraction_limited |= summary.extraction_limited;
        statistics->extraction_states += summary.extraction_states;
        statistics->extraction_us += summary.extraction_us;
        statistics->saturation_limited |=
            saturated.report.reason != eggc::StopReason::Saturated;
      }
      pruneDominatedStates(
          summary,
          [&](const TensorSharding& from, const TensorSharding& to,
              mlir::Type type) { return oracle.estimate(from, to, type); });
      if (observer.region_completed) observer.region_completed(summary);
      report.regions.push_back(std::move(summary));
    }
  }
  if (statistics) {
    statistics->evaluation_us =
        std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - evaluation_start)
            .count();
    ReshardPlanner planner = [&](auto from, auto to, auto type) {
      return oracle.plan(from, to, type);
    };
    if (report.chain) {
      auto& chain = *report.chain;
      chain.result =
          optimizeChain(report.regions, chain.interface, chain.contract,
                        planner, {options.max_chain_transitions});
      if (!chain.result.feasible)
        throw std::runtime_error("chain optimization failed: " +
                                 chain.result.failure);
      chain.result.lowered_mlir = materializeExecutionPlan(
          chain.result.execution, report.regions, report.mesh, model);
      if (observer.chain_completed) observer.chain_completed(chain);
    } else {
      auto& dag = *report.dag;
      dag.result = optimizeDag(report.regions, dag.interface, dag.contract,
                               planner, options.dag);
      if (!dag.result.feasible)
        throw std::runtime_error("DAG optimization failed: " +
                                 dag.result.failure);
      dag.result.lowered_mlir = materializeExecutionPlan(
          dag.result.execution, report.regions, report.mesh, model);
      if (observer.dag_completed) observer.dag_completed(dag);
    }
  }
  if (options.compose_regions) {
    auto [a, b] = *options.compose_regions;
    if (a >= report.regions.size() || b >= report.regions.size())
      throw std::invalid_argument("composition region index out of range");
    auto interface =
        buildPairInterface(source_regions[a], source_regions[b], values);
    auto pair = composePair(
        report.regions[a], report.regions[b], interface,
        [&](const TensorSharding& from, const TensorSharding& to,
            mlir::Type type) { return oracle.plan(from, to, type); },
        options.max_pair_evaluations);
    for (auto i : pair.frontier) {
      auto& plan = pair.plans[i];
      plan.lowered_mlir = materializePair(report.regions[a], report.regions[b],
                                          pair, i, report.mesh, model);
    }
    if (observer.pair_completed) observer.pair_completed(pair);
    report.compositions.push_back(std::move(pair));
  }
  return report;
}

}  // namespace joint_shard

#include "research/joint_shard/search/region_optimizer.h"

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
  if (!options.max_candidates || !options.max_boundary_states ||
      !options.max_pair_evaluations)
    throw std::invalid_argument("search caps must be positive");
  if (options.compose_pruned && !options.compose_regions)
    throw std::invalid_argument("--compose-pruned requires --compose-regions");
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
      summary.candidates = extractCandidates(saturated, options.rewriting,
                                             options.max_candidates);
      auto boundaries = enumerateBoundaryStates(
          region, report.mesh, options.layouts, options.max_boundary_states);
      summary.boundary_search_truncated = boundaries.truncated;
      summary.boundaries_evaluated = boundaries.states.size();
      if (boundaries.states.empty())
        summary.failures
            ["no compatible boundary layouts (check static shapes, mesh, and "
             "constraints)"] = 1;
      std::vector<mlir::OwningOpRef<mlir::ModuleOp>> prepared;
      for (const auto& candidate : summary.candidates) {
        auto candidateModule =
            prepareCandidateModule(region, candidate, report.mesh);
        if (observer.candidate_prepared)
          observer.candidate_prepared(region.id, candidate.id,
                                      *candidateModule);
        prepared.push_back(std::move(candidateModule));
      }
      for (size_t b = 0; b < boundaries.states.size(); ++b) {
        for (size_t c = 0; c < prepared.size(); ++c) {
          ShardyRunOptions runOptions;
          if (observer.snapshot) {
            runOptions.capture = SnapshotCapture::AllStages;
            runOptions.on_snapshot = [&](const ShardySnapshot& snapshot) {
              observer.snapshot(region.id, b, c, snapshot);
            };
          }
          summary.record(c, boundaries.states[b],
                         evaluator.evaluate(*prepared[c], boundaries.states[b],
                                            runOptions));
        }
      }
      summary.finalizePlans();
      pruneDominatedStates(
          summary,
          [&](const TensorSharding& from, const TensorSharding& to,
              mlir::Type type) { return oracle.estimate(from, to, type); });
      if (observer.region_completed) observer.region_completed(summary);
      report.regions.push_back(std::move(summary));
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
        options.max_pair_evaluations, options.compose_pruned);
    for (size_t i = 0; i < pair.plans.size(); ++i) {
      auto& plan = pair.plans[i];
      plan.lowered_mlir = materializePair(report.regions[a], report.regions[b],
                                          interface, plan, report.mesh, model);
    }
    if (observer.pair_completed) observer.pair_completed(pair);
    report.compositions.push_back(std::move(pair));
  }
  return report;
}

}  // namespace joint_shard

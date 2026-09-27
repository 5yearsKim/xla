#include "research/joint_shard/search/region_optimizer.h"

#include <sstream>
#include <stdexcept>

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Verifier.h"
#include "research/joint_shard/bridge/stablehlo_importer.h"

namespace {
const char* stopName(eggc::StopReason reason) {
  switch (reason) {
    case eggc::StopReason::Saturated:
      return "saturated";
    case eggc::StopReason::IterationLimit:
      return "iteration-limit";
    case eggc::StopReason::NodeLimit:
      return "node-limit";
    case eggc::StopReason::TimeLimit:
      return "time-limit";
    case eggc::StopReason::MatchLimit:
      return "match-limit";
    case eggc::StopReason::SearchLimit:
      return "search-limit";
    case eggc::StopReason::UserRequested:
      return "user-requested";
    case eggc::StopReason::MemoryLimit:
      return "memory-limit";
  }
  return "unknown";
}
void dumpModule(mlir::ModuleOp module, const std::string& path) {
  std::error_code error;
  llvm::raw_fd_ostream out(path, error);
  if (error)
    throw std::runtime_error("cannot write " + path + ": " + error.message());
  module.print(out);
  out << '\n';
  out.flush();
  if (out.has_error()) throw std::runtime_error("failed to write " + path);
}
}  // namespace

OptimizationReport summarizeRegions(mlir::ModuleOp module,
                                    const RegionOptimizerOptions& options) {
  if (mlir::failed(mlir::verify(module)))
    throw std::invalid_argument("invalid input module");
  if (!options.max_candidates || !options.max_boundary_states)
    throw std::invalid_argument("search caps must be positive");
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
      summary.id = region.id;
      summary.operations = region.operations.size();
      summary.inputs = region.inputs.size();
      summary.outputs = region.outputs.size();
      summary.oversized = region.oversized;
      for (auto value : region.inputs)
        summary.input_types.push_back(value.getType());
      for (auto value : region.outputs)
        summary.output_types.push_back(value.getType());
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
      std::string regionDirectory;
      if (!options.dump_directory.empty()) {
        regionDirectory =
            options.dump_directory + "/region_" + std::to_string(region.id);
        if (auto error = llvm::sys::fs::create_directories(regionDirectory))
          throw std::runtime_error(error.message());
      }
      for (const auto& candidate : summary.candidates) {
        auto candidateModule =
            prepareCandidateModule(region, candidate, report.mesh);
        if (!regionDirectory.empty())
          dumpModule(*candidateModule, regionDirectory + "/candidate_" +
                                           std::to_string(candidate.id) +
                                           ".mlir");
        prepared.push_back(std::move(candidateModule));
      }
      for (size_t b = 0; b < boundaries.states.size(); ++b) {
        for (size_t c = 0; c < prepared.size(); ++c) {
          ShardyRunOptions runOptions;
          if (!regionDirectory.empty())
            runOptions.dumpDirectory = regionDirectory + "/boundary_" +
                                       std::to_string(b) + "/candidate_" +
                                       std::to_string(c);
          summary.record(c, boundaries.states[b],
                         evaluator.evaluate(*prepared[c], boundaries.states[b],
                                            runOptions));
        }
      }
      summary.keepBestPerBoundary();
      pruneDominatedStates(
          summary,
          [&](const TensorSharding& from, const TensorSharding& to,
              mlir::Type type) { return oracle.estimate(from, to, type); });
      if (!regionDirectory.empty()) {
        std::error_code error;
        llvm::raw_fd_ostream out(regionDirectory + "/summary.txt", error);
        if (error) throw std::runtime_error(error.message());
        out << summary.str();
        out.flush();
        if (out.has_error())
          throw std::runtime_error("failed to write summary");
        for (size_t p = 0; p < summary.plans.size(); ++p) {
          llvm::raw_fd_ostream artifact(
              regionDirectory + "/frontier_" + std::to_string(p) + ".mlir",
              error);
          if (error) throw std::runtime_error(error.message());
          artifact << summary.plans[p].lowered_mlir;
          artifact.flush();
          if (artifact.has_error())
            throw std::runtime_error("failed to write frontier");
        }
      }
      report.regions.push_back(std::move(summary));
    }
  }
  return report;
}

std::string OptimizationReport::str() const {
  std::ostringstream out;
  out << "Joint rewrite-sharding region summaries\nMesh: " << mesh.name
      << " devices=" << mesh.mesh.getTotalSize()
      << "\nCosts are illustrative estimates in microseconds.\n";
  for (const auto& [reason, count] : preserved_operations)
    out << "Preserved boundary: " << reason << " count=" << count << '\n';
  for (size_t i = 0; i < regions.size(); ++i) {
    out << '\n' << regions[i].str();
    const auto& run = saturation[i];
    out << "Saturation iterations=" << run.iterations << " nodes=" << run.nodes
        << " stop=" << stopName(run.reason) << '\n';
  }
  return out.str();
}

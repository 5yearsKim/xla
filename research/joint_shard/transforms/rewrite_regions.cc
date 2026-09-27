#include "research/joint_shard/transforms/rewrite_regions.h"

#include <iomanip>
#include <limits>
#include <sstream>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "research/joint_shard/bridge/stablehlo_exporter.h"
#include "research/joint_shard/bridge/stablehlo_importer.h"
#include "research/joint_shard/transforms/region_candidates.h"

namespace {
const char* stopName(eggc::StopReason reason);
void rewriteIsland(const std::vector<mlir::Operation*>& island,
                   const CompiledTensorRules& rules,
                   const TensorRewriteOptions& options,
                   TensorRewriteReport& report) {
  if (island.empty()) return;
  auto region = describeRegion(island);
  region.id = report.regions++;
  report.imported_operations += island.size();
  report.roots += region.outputs.size();
  if (!region.outputs.empty()) {
    auto saturated = saturateRegion(region, rules, options);
    report.runs.push_back(saturated.report);
    TensorExtractionReport extraction;
    auto candidate = mergeExtractedRoots(
        extractTensorRoots(*saturated.graph, saturated.roots,
                           options.extraction, options.extractor, options.dag,
                           extraction),
        "selected");
    report.extractions.push_back(std::move(extraction));
    mlir::OpBuilder builder(island.front());
    StableHloExporter exporter(builder, island.front()->getLoc(),
                               region.inputs);
    auto values =
        exporter.exportRoots(candidate.expression, candidate.output_roots);
    for (size_t i = 0; i < region.outputs.size(); ++i)
      region.outputs[i].replaceAllUsesWith(values[i]);
  }
  for (auto it = island.rbegin(); it != island.rend(); ++it)
    if ((*it)->use_empty()) (*it)->erase();
}
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
const char* dagStopName(eggc::DagStopReason reason) {
  switch (reason) {
    case eggc::DagStopReason::Exhausted:
      return "exhausted";
    case eggc::DagStopReason::StateLimit:
      return "state-limit";
    case eggc::DagStopReason::TimeLimit:
      return "time-limit";
    case eggc::DagStopReason::FrontierLimit:
      return "frontier-limit";
  }
  return "unknown";
}
}  // namespace
mlir::LogicalResult rewriteUnconstrainedRegions(
    mlir::ModuleOp module, const TensorRewriteOptions& options,
    TensorRewriteReport* output) {
  for (auto function : module.getOps<mlir::func::FuncOp>())
    if (!function.isExternal() && !llvm::hasSingleElement(function.getBody()))
      return function.emitError(
          "region rewriting requires single-block functions");
  // Parse and validate once, before changing the module.
  auto semantic = options.semantic;
  if (!semantic.report)
    semantic.report =
        std::make_shared<std::map<std::string, SemanticRuleStats>>();
  auto compileOptions = options;
  compileOptions.semantic = semantic;
  auto compiledRules = compileTensorRules(compileOptions);
  TensorRewriteReport report;
  for (auto function : module.getOps<mlir::func::FuncOp>()) {
    if (function.isExternal()) continue;
    std::vector<mlir::Operation*> original;
    for (auto& op : function.getBody().front()) original.push_back(&op);
    std::vector<mlir::Operation*> island;
    for (auto* op : original) {
      if (canImportTensorOperation(op)) {
        island.push_back(op);
        continue;
      }
      rewriteIsland(island, compiledRules, options, report);
      island.clear();
      ++report.boundaries[op->getName().getStringRef().str() + ": " +
                          tensorImportRejection(op)];
    }
    rewriteIsland(island, compiledRules, options, report);
  }
  report.rules = *semantic.report;
  if (output) *output = std::move(report);
  return mlir::verify(module);
}
std::string TensorRewriteReport::str() const {
  std::ostringstream out;
  out << "regions=" << regions << " imported=" << imported_operations
      << " roots=" << roots << '\n';
  for (const auto& [name, count] : boundaries)
    out << "boundary " << name << " count=" << count << '\n';
  for (const auto& [name, stats] : rules) {
    out << "rule " << name << " visits=" << stats.node_visits
        << " matches=" << stats.structural_matches
        << " accepted=" << stats.accepted << " applied=" << stats.applied
        << " duplicates=" << stats.duplicates
        << " budget_stops=" << stats.budget_stops << '\n';
    for (const auto& [reason, count] : stats.rejections)
      out << "  rejected " << reason << " count=" << count << '\n';
  }
  for (size_t i = 0; i < runs.size(); ++i)
    out << "run " << i << " stop=" << stopName(runs[i].reason)
        << " iterations=" << runs[i].iterations << " nodes=" << runs[i].nodes
        << '\n';
  for (size_t i = 0; i < extractions.size(); ++i) {
    const auto& extraction = extractions[i];
    out << "extraction " << i
        << " selected=" << (extraction.selected_dag ? "dag" : "tree")
        << " roots=" << extraction.roots
        << " attempted_dag=" << extraction.attempted_dag;
    if (extraction.stop)
      out << " stop=" << dagStopName(*extraction.stop)
          << " optimal=" << extraction.optimal
          << " states=" << extraction.explored_states
          << " peak_frontier=" << extraction.peak_frontier << " search_us="
          << std::chrono::duration_cast<std::chrono::microseconds>(
                 extraction.search_time)
                 .count();
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (extraction.baseline_cost)
      out << " baseline_cost=" << *extraction.baseline_cost;
    if (extraction.candidate_cost)
      out << " candidate_cost=" << *extraction.candidate_cost;
    if (!extraction.fallback.empty())
      out << " fallback=" << extraction.fallback;
    out << '\n';
  }
  return out.str();
}

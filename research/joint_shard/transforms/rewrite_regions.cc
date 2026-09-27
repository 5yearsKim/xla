#include "research/joint_shard/transforms/rewrite_regions.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "research/joint_shard/bridge/stablehlo_exporter.h"
#include "research/joint_shard/bridge/stablehlo_importer.h"
#include "research/joint_shard/transforms/region_candidates.h"

namespace joint_shard {

namespace {
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

}  // namespace joint_shard

#include "research/joint_shard/transforms/rewrite_regions.h"

#include <algorithm>
#include <charconv>
#include <iomanip>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "research/joint_shard/bridge/stablehlo_exporter.h"
#include "research/joint_shard/bridge/stablehlo_importer.h"

namespace {
const char* stopName(eggc::StopReason reason);
struct NodeHash {
  size_t operator()(const TensorNode& node) const { return node.hash(); }
};
void rewriteIsland(const std::vector<mlir::Operation*>& island,
                   const eggc::CompiledRules<TensorNode, TensorAnalysis>& rules,
                   const TensorRewriteOptions& options,
                   TensorRewriteReport& report) {
  if (island.empty()) return;
  llvm::DenseSet<mlir::Operation*> members;
  for (auto* op : island) members.insert(op);
  std::vector<mlir::Value> roots, leaves;
  llvm::DenseSet<mlir::Value> leafSet;
  for (auto* op : island) {
    for (auto operand : op->getOperands())
      if (!members.contains(operand.getDefiningOp()) &&
          leafSet.insert(operand).second)
        leaves.push_back(operand);
    auto value = op->getResult(0);
    for (auto& use : value.getUses())
      if (!members.contains(use.getOwner())) {
        roots.push_back(value);
        break;
      }
  }
  ++report.regions;
  report.imported_operations += island.size();
  report.roots += roots.size();
  if (!roots.empty()) {
    TensorEGraph graph(TensorAnalysis{});
    StableHloImporter importer(graph);
    for (unsigned i = 0; i < leaves.size(); ++i)
      importer.bindValue(leaves[i], i);
    std::vector<eggc::Id> ids;
    for (auto root : roots) ids.push_back(importer.importValue(root));
    report.runs.push_back(eggc::run(graph, rules, options.runner));
    if (options.print_egraph) {
      graph.rebuild();
      auto& out = llvm::errs();
      out << "egraph region=" << report.regions - 1
          << " stop=" << stopName(report.runs.back().reason) << " roots=";
      for (auto id : ids) out << " e" << graph.find(id);
      out << '\n';
      for (auto id : graph.classes()) {
        out << "e" << id << ":\n";
        for (const auto& node : graph.nodes(id)) {
          out << "  " << node.format();
          for (auto child : node.children()) out << " e" << graph.find(child);
          out << '\n';
        }
      }
    }
    TensorExtractionReport extraction;
    auto selectedRoots =
        extractTensorRoots(graph, ids, options.extraction, options.extractor,
                           options.dag, extraction);
    report.extractions.push_back(std::move(extraction));
    TensorRecExpr expression;
    std::unordered_map<TensorNode, eggc::Id, NodeHash> shared;
    std::vector<size_t> outputIds;
    for (const auto& selected : selectedRoots) {
      std::vector<eggc::Id> remapped;
      for (auto node : selected.nodes) {
        for (auto& operand : node.operands) operand = remapped[operand];
        auto found = shared.find(node);
        if (found != shared.end())
          remapped.push_back(found->second);
        else {
          auto output = expression.add(node);
          shared.emplace(std::move(node), output);
          remapped.push_back(output);
        }
      }
      outputIds.push_back(remapped.back());
    }
    mlir::OpBuilder builder(island.front());
    StableHloExporter exporter(builder, island.front()->getLoc(),
                               std::move(leaves));
    auto values = exporter.exportRoots(expression, outputIds);
    for (size_t i = 0; i < roots.size(); ++i)
      roots[i].replaceAllUsesWith(values[i]);
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
size_t positiveNumber(std::string_view text) {
  size_t value = 0;
  auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size() || !value)
    throw std::invalid_argument("expected a positive limit: " +
                                std::string(text));
  return value;
}
std::chrono::milliseconds positiveMilliseconds(std::string_view text) {
  auto value = positiveNumber(text);
  using Rep = std::chrono::milliseconds::rep;
  const auto max = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::duration::max())
                       .count();
  if (value > static_cast<size_t>(max))
    throw std::invalid_argument("time limit is too large: " +
                                std::string(text));
  return std::chrono::milliseconds(static_cast<Rep>(value));
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
  auto rules = options.rules_file.empty()
                   ? parseSemanticRules(defaultSemanticRules(),
                                        options.numerical_policy, semantic)
                   : loadSemanticRulesFile(options.rules_file,
                                           options.numerical_policy, semantic);
  auto attributeRules = buildAttributeRewrites();
  rules.insert(rules.end(), attributeRules.begin(), attributeRules.end());
  std::unordered_set<std::string> ruleNames;
  for (const auto& rule : rules)
    if (!ruleNames.insert(rule.name).second)
      throw std::invalid_argument("duplicate DSL/C++ rule name: " + rule.name);
  // Freeze and validate the rules once for the module. Ordinary pattern rules
  // can also reuse their compiled matcher and replacement programs per island.
  eggc::CompiledRules<TensorNode, TensorAnalysis> compiledRules(
      std::move(rules));
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
std::string_view tensorRewriteOptionHelp() {
  return "[--rules=path] [--numerical-policy=strict|relaxed] [--iterations=N] "
         "[--nodes=N] [--matches=N] [--search-visits=N] [--time-ms=N] "
         "[--extraction=compute|depth|memory] [--rule-group=exact|algebra|all] "
         "[--extractor=auto|tree] [--dag-states=N] [--dag-time-ms=N] "
         "[--dag-frontier=N] "
         "[--rewrite-report] [--print-egraph]";
}
bool parseTensorRewriteOption(std::string_view argument,
                              TensorRewriteOptions& options) {
  if (argument == "--print-egraph") {
    options.print_egraph = true;
    return true;
  }
  bool NumericalPermissions::* permission = nullptr;
  if (argument == "--allow-fp-reorder")
    permission = &NumericalPermissions::reorder_floating_point;
  else if (argument == "--allow-fp-reassociate")
    permission = &NumericalPermissions::reassociate_floating_point;
  else if (argument == "--allow-fp-distribute")
    permission = &NumericalPermissions::distribute_floating_point;
  else if (argument == "--assume-finite")
    permission = &NumericalPermissions::assume_finite;
  else if (argument == "--ignore-signed-zero")
    permission = &NumericalPermissions::ignore_signed_zero;
  else if (argument == "--allow-dot-arithmetic")
    permission = &NumericalPermissions::rewrite_dot_arithmetic;
  if (permission) {
    if (!options.semantic.permissions)
      options.semantic.permissions =
          numericalPermissions(options.numerical_policy);
    (*options.semantic.permissions).*permission = true;
    return true;
  }
  auto at = argument.find('=');
  if (at == std::string_view::npos) return false;
  auto name = argument.substr(0, at), value = argument.substr(at + 1);
  if (name == "--rules") {
    if (value.empty()) throw std::invalid_argument("empty rule path");
    options.rules_file = value;
  } else if (name == "--numerical-policy") {
    if (value == "strict")
      options.numerical_policy = NumericalPolicy::PreserveEvaluation;
    else if (value == "relaxed")
      options.numerical_policy = NumericalPolicy::AllowReassociation;
    else
      throw std::invalid_argument("numerical policy must be strict or relaxed");
    options.semantic.permissions.reset();
  } else if (name == "--rule-group") {
    if (value != "exact" && value != "algebra" && value != "all")
      throw std::invalid_argument("rule group must be exact, algebra, or all");
    options.semantic.enable_associativity = value != "exact";
    options.semantic.enable_linearity = value == "all";
    options.semantic.enable_homogeneity = value == "all";
  } else if (name == "--iterations")
    options.runner.iteration_limit = positiveNumber(value);
  else if (name == "--nodes")
    options.runner.node_limit = positiveNumber(value);
  else if (name == "--matches") {
    options.runner.match_limit = positiveNumber(value);
    options.semantic.match_limit = positiveNumber(value);
  } else if (name == "--search-visits")
    options.semantic.visit_limit = positiveNumber(value);
  else if (name == "--time-ms")
    options.runner.time_limit = positiveMilliseconds(value);
  else if (name == "--extractor") {
    if (value == "auto")
      options.extractor = TensorExtractorMode::Auto;
    else if (value == "tree")
      options.extractor = TensorExtractorMode::Tree;
    else
      throw std::invalid_argument("extractor must be auto or tree");
  } else if (name == "--dag-states")
    options.dag.state_limit = positiveNumber(value);
  else if (name == "--dag-time-ms")
    options.dag.time_limit = positiveMilliseconds(value);
  else if (name == "--dag-frontier")
    options.dag.frontier_limit = positiveNumber(value);
  else if (name == "--extraction") {
    if (value == "compute")
      options.extraction = ExtractionProfile::Compute;
    else if (value == "depth")
      options.extraction = ExtractionProfile::Depth;
    else if (value == "memory")
      options.extraction = ExtractionProfile::Memory;
    else
      throw std::invalid_argument("unknown extraction profile");
  } else
    return false;
  return true;
}

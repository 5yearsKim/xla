#include "research/joint_shard/reporting/reports.h"

#include <chrono>
#include <iomanip>
#include <limits>
#include <sstream>

#include "llvm/Support/raw_ostream.h"
#include "research/joint_shard/search/region_optimizer.h"
#include "research/joint_shard/transforms/rewrite_regions.h"

namespace joint_shard {
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
std::string typeText(mlir::Type type) {
  std::string text;
  llvm::raw_string_ostream out(text);
  out << type;
  return text;
}
std::string compact(const TensorSharding& layout) {
  std::ostringstream out;
  bool replicated = true;
  auto dims = layout.attr.getDimShardings();
  for (size_t i = 0; i < dims.size(); ++i)
    for (auto axis : dims[i].getAxes()) {
      if (!replicated) out << ",";
      out << axis.getName().str() << ":d" << i;
      replicated = false;
    }
  return replicated ? "R" : out.str();
}
}  // namespace
std::string formatReport(const RegionSummary& report) {
  std::ostringstream out;
  out << "Region " << report.id
      << "\n========\nOperations: " << report.operations
      << "\nInputs: " << report.interface.inputs.size()
      << "\nOutputs: " << report.interface.outputs.size()
      << "\nCandidates extracted: " << report.candidates.size()
      << "\nBoundary states evaluated: " << report.boundaries_evaluated
      << "\nEvaluations: " << report.evaluations
      << "\nBefore best-per-boundary: " << report.feasible_plans << " plans"
      << "\nAfter best-per-boundary: " << report.best_boundary_plans << " plans"
      << "\nAfter reshard dominance: " << report.frontier.size() << " plans"
      << "\nUnknown-cost evaluations: " << report.unknown_cost_plans
      << "\nOversized protected region: " << report.oversized
      << "\nBoundary search truncated: " << report.boundary_search_truncated
      << '\n';
  for (const auto& [reason, count] : report.failures)
    out << "Rejected: " << reason << " count=" << count << '\n';
  for (const auto& candidate : report.candidates)
    out << "Candidate R" << candidate.id << ": " << candidate.name << '\n';
  out << "\nBoundary | Rewrite | Compute(us) | Comm(us) | Total(us)\n";
  out << std::setprecision(12);
  for (auto id : report.frontier) {
    const auto& plan = report.plans.at(id);
    out << "P" << id << " " << plan.boundary.key() << " | R"
        << plan.candidate_id << " | ";
    if (plan.cost.known())
      out << plan.cost.compute << " | " << plan.cost.communication << " | "
          << plan.cost.total();
    else
      out << "unknown | unknown | unknown";
    out << '\n';
  }
  for (const auto& witness : report.dominance)
    out << "Pruned P" << witness.removed_id << " " << witness.removed.key()
        << " via P" << witness.replacement_id << " "
        << witness.replacement.key()
        << " adapters_us=" << witness.adapters.total()
        << " replacement_total_us=" << witness.replacement_total << '\n';
  return out.str();
}
std::string formatReport(const PairSummary& report) {
  std::ostringstream out;
  out << "\nComposition " << report.a_region << " -> " << report.b_region
      << " (" << (report.frontier_resolved ? "frontier-resolved" : "exact")
      << ")\n"
      << "Pairs evaluated: " << report.evaluations
      << "; truncated: " << report.truncated
      << "; shared layout rejections: " << report.shared_layout_rejections
      << "; unknown cost rejections: " << report.unknown_cost_rejections << "\n"
      << "External boundary winners: " << report.plans.size() << "\n"
      << "Layouts: R=replicated, axis:dN=axis shards tensor dimension N. Costs "
         "in us.\n";
  for (size_t i = 0; i < report.interface.external.inputs.size(); ++i) {
    auto type = typeText(report.interface.external.inputs[i].type);
    out << "Input " << i << " = V" << report.interface.external.inputs[i].value
        << " " << type << "\n";
  }
  for (size_t i = 0; i < report.interface.external.outputs.size(); ++i)
    out << "Output " << i << " = V"
        << report.interface.external.outputs[i].value << " "
        << typeText(report.interface.external.outputs[i].type) << "\n";
  for (size_t i = 0; i < report.interface.a_inputs.size(); ++i)
    out << "A input " << i << " <- external input "
        << report.interface.a_inputs[i] << "\n";
  for (size_t i = 0; i < report.interface.b_inputs.size(); ++i) {
    out << "B input " << i << " <- ";
    if (report.interface.b_inputs[i] < 0)
      out << "A output 0";
    else
      out << "external input " << report.interface.b_inputs[i];
    out << "\n";
  }
  out << "Intermediate V" << report.interface.intermediate.value << " "
      << typeText(report.interface.intermediate.type)
      << " maps A output 0 -> B input " << report.interface.intermediate_input
      << "\n";
  out << std::setprecision(12);
  for (size_t i = 0; i < report.plans.size(); ++i) {
    const auto& p = report.plans[i];
    out << "Selected " << i << ": inputs=[";
    for (size_t j = 0; j < p.boundary.inputs.size(); ++j)
      out << (j ? "," : "") << compact(p.boundary.inputs[j]);
    out << "] outputs=[";
    for (size_t j = 0; j < p.boundary.outputs.size(); ++j)
      out << (j ? "," : "") << compact(p.boundary.outputs[j]);
    out << "] A=P" << p.a.requested << "/implemented P" << p.a.implementation
        << "(R" << p.a.candidate_id << ") B=P" << p.b.requested
        << "/implemented P" << p.b.implementation << "(R" << p.b.candidate_id
        << ") intermediate=" << compact(p.intermediate.from) << " -> "
        << compact(p.intermediate.to) << " A=" << p.a.cost.total()
        << " (core=" << p.a.core_cost.total()
        << ",wrappers=" << p.a.adapters_cost.total()
        << ") adapter=" << p.intermediate.cost.total()
        << " B=" << p.b.cost.total() << " (core=" << p.b.core_cost.total()
        << ",wrappers=" << p.b.adapters_cost.total()
        << ") total=" << p.cost.total() << " compute=" << p.cost.compute
        << " comm=" << p.cost.communication << "\n";
  }
  return out.str();
}
std::string formatReport(const OptimizationReport& report) {
  std::ostringstream out;
  out << "Joint rewrite-sharding region summaries\nMesh: " << report.mesh.name
      << " devices=" << report.mesh.mesh.getTotalSize()
      << "\nCosts are illustrative estimates in microseconds.\n";
  for (const auto& [reason, count] : report.preserved_operations)
    out << "Preserved boundary: " << reason << " count=" << count << '\n';
  for (size_t i = 0; i < report.regions.size(); ++i) {
    out << '\n' << formatReport(report.regions[i]);
    const auto& run = report.saturation[i];
    out << "Saturation iterations=" << run.iterations << " nodes=" << run.nodes
        << " stop=" << stopName(run.reason) << '\n';
  }
  for (const auto& pair : report.compositions) out << formatReport(pair);
  return out.str();
}
std::string formatReport(const TensorRewriteReport& report) {
  std::ostringstream out;
  out << "regions=" << report.regions
      << " imported=" << report.imported_operations << " roots=" << report.roots
      << '\n';
  for (const auto& [name, count] : report.boundaries)
    out << "boundary " << name << " count=" << count << '\n';
  for (const auto& [name, stats] : report.rules) {
    out << "rule " << name << " visits=" << stats.node_visits
        << " matches=" << stats.structural_matches
        << " accepted=" << stats.accepted << " applied=" << stats.applied
        << " duplicates=" << stats.duplicates
        << " budget_stops=" << stats.budget_stops << '\n';
    for (const auto& [reason, count] : stats.rejections)
      out << "  rejected " << reason << " count=" << count << '\n';
  }
  for (size_t i = 0; i < report.runs.size(); ++i)
    out << "run " << i << " stop=" << stopName(report.runs[i].reason)
        << " iterations=" << report.runs[i].iterations
        << " nodes=" << report.runs[i].nodes << '\n';
  for (size_t i = 0; i < report.extractions.size(); ++i) {
    const auto& extraction = report.extractions[i];
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

}  // namespace joint_shard

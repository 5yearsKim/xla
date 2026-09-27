#include "research/joint_shard/transforms/tensor_extraction.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>

namespace {
size_t saturatedMultiply(size_t a, size_t b) {
  const auto max = std::numeric_limits<size_t>::max() / 4;
  return b && a > max / b ? max : a * b;
}

size_t localCost(const TensorNode& node, std::span<const TensorFacts> operands,
                 const InferenceResult& inferred, ExtractionProfile profile) {
  size_t local = 1;
  if (inferred.valid() && node.op != OpKind::Input &&
      node.op != OpKind::Constant) {
    auto type = inferred.facts.type;
    local = type.hasStaticShape() ? std::max<int64_t>(1, type.getNumElements())
                                  : 1024;
    if (profile == ExtractionProfile::Compute) {
      if (node.op == OpKind::DotGeneral) {
        const auto& attrs = std::get<DotGeneralAttrs>(node.attrs);
        for (auto axis : attrs.lhs_contracting) {
          auto size = operands[0].type.getDimSize(axis);
          local = saturatedMultiply(local,
                                    size < 0 ? 32 : std::max<int64_t>(1, size));
        }
        local = saturatedMultiply(local, 2);
      } else if (node.op == OpKind::Exp || node.op == OpKind::Log ||
                 node.op == OpKind::Sqrt || node.op == OpKind::Tanh) {
        local = saturatedMultiply(local, 8);
      } else if (node.op == OpKind::Transpose || node.op == OpKind::Reshape ||
                 node.op == OpKind::BroadcastInDim) {
        local = 1;
      }
    }
    if (profile == ExtractionProfile::Depth) local = 1;
  }
  return local;
}

size_t graphLocalCost(const TensorEGraph& graph, const TensorNode& node,
                      ExtractionProfile profile) {
  std::vector<TensorFacts> operands;
  for (auto id : node.operands) operands.push_back(tensorFacts(graph, id));
  return localCost(node, operands, inferTensorNode(node, operands), profile);
}

eggc::CostPolicy<TensorNode> treeCost(const TensorEGraph& graph,
                                      ExtractionProfile profile) {
  return [&graph, profile](
             const TensorNode& node,
             const std::vector<size_t>& children) -> std::optional<size_t> {
    size_t total = graphLocalCost(graph, node, profile);
    for (auto cost : children) {
      if (profile == ExtractionProfile::Depth) {
        if (cost == std::numeric_limits<size_t>::max()) return std::nullopt;
        total = std::max(total, cost + 1);
      } else {
        if (cost > std::numeric_limits<size_t>::max() - total)
          return std::nullopt;
        total += cost;
      }
    }
    return total;
  };
}

// RecExpr operands index earlier expression nodes, not graph e-classes.
// Infer facts bottom-up so costs never accidentally query unrelated classes.
std::optional<double> expressionCost(const TensorRecExpr& expression) {
  if (expression.nodes.empty()) return std::nullopt;
  std::vector<TensorFacts> facts;
  double total = 0;
  for (const auto& node : expression.nodes) {
    std::vector<TensorFacts> operands;
    for (auto child : node.operands) {
      if (child >= facts.size())
        throw std::logic_error("extracted children must precede parents");
      operands.push_back(facts[child]);
    }
    auto inferred = inferTensorNode(node, operands);
    if (inferred.status == InferenceStatus::Invalid) return std::nullopt;
    total += static_cast<double>(
        localCost(node, operands, inferred, ExtractionProfile::Compute));
    if (!std::isfinite(total)) return std::nullopt;
    facts.push_back(inferred.facts);
  }
  return total;
}
}  // namespace

std::vector<TensorRecExpr> extractTensorRoots(
    const TensorEGraph& graph, const std::vector<eggc::Id>& roots,
    ExtractionProfile profile, TensorExtractorMode mode,
    const eggc::DagOptions& budgets, TensorExtractionReport& report) {
  report = {};
  report.roots = roots.size();
  TensorExtractor baseline(graph, roots, treeCost(graph, profile));
  std::vector<TensorRecExpr> expressions;
  for (auto root : roots)
    expressions.push_back(baseline.find_best(root).second);
  if (roots.size() == 1 && profile == ExtractionProfile::Compute)
    report.baseline_cost = expressionCost(expressions.front());
  if (roots.empty()) {
    report.fallback = "no-outputs";
  } else if (mode == TensorExtractorMode::Tree) {
    report.fallback = "tree-requested";
  } else if (roots.size() != 1) {
    report.fallback = "multiple-outputs";
  } else if (profile != ExtractionProfile::Compute) {
    report.fallback = "non-compute-profile";
  } else if (!report.baseline_cost) {
    report.fallback = "unavailable-baseline-cost";
  } else {
    report.attempted_dag = true;
    eggc::DagExtractor<TensorNode, TensorAnalysis> extractor(
        graph, [&graph](const TensorNode& node) -> std::optional<double> {
          return static_cast<double>(
              graphLocalCost(graph, node, ExtractionProfile::Compute));
        });
    auto start = std::chrono::steady_clock::now();
    auto candidate = extractor.solve(roots.front(), budgets);
    report.search_time = std::chrono::steady_clock::now() - start;
    report.stop = candidate.reason;
    report.optimal = candidate.optimal;
    report.explored_states = candidate.explored_states;
    report.peak_frontier = candidate.peak_frontier;
    if (!candidate.cost) {
      report.fallback = "no-dag-candidate";
    } else {
      report.candidate_cost = expressionCost(candidate.expression);
      if (!report.candidate_cost) {
        report.fallback = "unavailable-candidate-cost";
      } else if (*report.candidate_cost < *report.baseline_cost) {
        expressions.front() = std::move(candidate.expression);
        report.selected_dag = true;
      } else {
        report.fallback = "baseline-no-worse";
      }
    }
  }
  return expressions;
}

#include "research/joint_shard/transforms/region_candidates.h"

#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "llvm/Support/raw_ostream.h"
#include "research/joint_shard/bridge/stablehlo_importer.h"

namespace joint_shard {

namespace {
struct NodeHash {
  size_t operator()(const TensorNode& node) const { return node.hash(); }
};
// Canonicalize reachable nodes in output order, including attrs and sharing.
// Input nodes unused by this expression do not affect structural equality.
Candidate canonicalize(const Candidate& candidate) {
  Candidate result;
  std::unordered_map<TensorNode, eggc::Id, NodeHash> shared;
  std::unordered_map<size_t, eggc::Id> remapped;
  std::function<eggc::Id(size_t)> visit = [&](size_t index) {
    if (auto found = remapped.find(index); found != remapped.end())
      return found->second;
    auto node = candidate.expression.nodes.at(index);
    for (auto& operand : node.operands) operand = visit(operand);
    auto [entry, inserted] =
        shared.emplace(node, result.expression.nodes.size());
    if (inserted) result.expression.add(std::move(node));
    return remapped[index] = entry->second;
  };
  for (auto root : candidate.output_roots)
    result.output_roots.push_back(visit(root));
  return result;
}
}  // namespace

CompiledTensorRules compileTensorRules(const TensorRewriteOptions& options) {
  auto rules =
      options.rules_file.empty()
          ? parseSemanticRules(defaultSemanticRules(), options.numerical_policy,
                               options.semantic)
          : loadSemanticRulesFile(options.rules_file, options.numerical_policy,
                                  options.semantic);
  auto attributeRules = buildAttributeRewrites();
  rules.insert(rules.end(), attributeRules.begin(), attributeRules.end());
  std::unordered_set<std::string> names;
  for (const auto& rule : rules)
    if (!names.insert(rule.name).second)
      throw std::invalid_argument("duplicate DSL/C++ rule name: " + rule.name);
  return CompiledTensorRules(std::move(rules));
}

SaturatedRegion saturateRegion(const Region& region,
                               const CompiledTensorRules& rules,
                               const TensorRewriteOptions& options) {
  SaturatedRegion result;
  result.graph = std::make_unique<TensorEGraph>(TensorAnalysis{});
  StableHloImporter importer(*result.graph);
  for (unsigned i = 0; i < region.inputs.size(); ++i)
    importer.bindValue(region.inputs[i], i);
  for (auto value : region.outputs) {
    result.roots.push_back(importer.importValue(value));
    result.original.output_roots.push_back(importer.originalRoot(value));
  }
  result.original.name = "original";
  result.original.expression = importer.originalExpression();
  result.report = eggc::run(*result.graph, rules, options.runner);
  if (options.print_egraph) {
    auto& out = llvm::errs();
    out << "egraph region=" << region.id << " roots=";
    for (auto id : result.roots) out << " e" << result.graph->find(id);
    out << '\n';
    for (auto id : result.graph->classes()) {
      out << "e" << id << ":\n";
      for (const auto& node : result.graph->nodes(id)) {
        out << "  " << node.format();
        for (auto child : node.children())
          out << " e" << result.graph->find(child);
        out << '\n';
      }
    }
  }
  return result;
}

Candidate mergeExtractedRoots(const std::vector<TensorRecExpr>& roots,
                              std::string name) {
  Candidate candidate;
  candidate.name = std::move(name);
  std::unordered_map<TensorNode, eggc::Id, NodeHash> shared;
  for (const auto& root : roots) {
    std::vector<eggc::Id> remapped;
    for (auto node : root.nodes) {
      for (auto& operand : node.operands) operand = remapped.at(operand);
      auto [entry, inserted] =
          shared.emplace(node, candidate.expression.nodes.size());
      if (inserted) candidate.expression.add(std::move(node));
      remapped.push_back(entry->second);
    }
    if (remapped.empty()) throw std::invalid_argument("empty extracted root");
    candidate.output_roots.push_back(remapped.back());
  }
  return candidate;
}

std::vector<Candidate> extractCandidates(
    const SaturatedRegion& saturated, const TensorRewriteOptions& options,
    size_t max_candidates, CandidateExtractionReport* diagnostics) {
  if (!max_candidates)
    throw std::invalid_argument("candidate cap must be positive");
  std::vector<Candidate> candidates{saturated.original};
  if (diagnostics) *diagnostics = {};
  std::vector<Candidate> identities{canonicalize(saturated.original)};
  auto append = [&](ExtractionProfile profile, TensorExtractorMode mode,
                    std::string name) {
    if (candidates.size() >= max_candidates) {
      if (diagnostics) diagnostics->profiles_skipped_at_cap = true;
      return;
    }
    TensorExtractionReport report;
    auto candidate = mergeExtractedRoots(
        extractTensorRoots(*saturated.graph, saturated.roots, profile, mode,
                           options.dag, report),
        std::move(name));
    if (diagnostics) diagnostics->profiles.push_back(report);
    auto identity = canonicalize(candidate);
    for (const auto& previous : identities)
      if (previous.output_roots == identity.output_roots &&
          previous.expression.nodes == identity.expression.nodes)
        return;
    candidate.id = candidates.size();
    identities.push_back(std::move(identity));
    candidates.push_back(std::move(candidate));
  };
  const char* firstName =
      options.extraction == ExtractionProfile::Compute ? "compute-tree"
      : options.extraction == ExtractionProfile::Depth ? "depth"
                                                       : "memory";
  append(options.extraction, TensorExtractorMode::Tree, firstName);
  append(ExtractionProfile::Compute, TensorExtractorMode::Tree, "compute-tree");
  append(ExtractionProfile::Compute, options.extractor, "compute-dag");
  append(ExtractionProfile::Depth, TensorExtractorMode::Tree, "depth");
  append(ExtractionProfile::Memory, TensorExtractorMode::Tree, "memory");
  return candidates;
}

}  // namespace joint_shard

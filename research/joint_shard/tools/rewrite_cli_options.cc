#include "research/joint_shard/tools/rewrite_cli_options.h"

#include <charconv>
#include <stdexcept>

namespace joint_shard {

namespace {
bool parseBoolean(std::string_view value) {
  if (value == "true") return true;
  if (value == "false") return false;
  throw std::invalid_argument("expected true or false");
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

void addRewriteCliOptions(cxxopts::Options& options) {
  options.add_options("Rewrite")("rules", "Path to a tensor rewrite rules file",
                                 cxxopts::value<std::string>())(
      "numerical-policy", "Floating-point policy: strict or relaxed",
      cxxopts::value<std::string>())("iterations", "Rewrite iteration limit",
                                     cxxopts::value<std::string>())(
      "nodes", "E-graph node limit", cxxopts::value<std::string>())(
      "matches", "Rewrite match limit", cxxopts::value<std::string>())(
      "search-visits", "Semantic search visit limit",
      cxxopts::value<std::string>())("time-ms",
                                     "Rewrite time limit in milliseconds",
                                     cxxopts::value<std::string>())(
      "extraction", "Extraction profile: compute, depth, or memory",
      cxxopts::value<std::string>())("rule-group",
                                     "Rule group: exact, algebra, or all",
                                     cxxopts::value<std::string>())(
      "extractor", "Extractor: auto or tree", cxxopts::value<std::string>())(
      "dag-states", "DAG extraction state limit",
      cxxopts::value<std::string>())(
      "dag-time-ms", "DAG extraction time limit in milliseconds",
      cxxopts::value<std::string>())("dag-frontier",
                                     "DAG extraction frontier limit",
                                     cxxopts::value<std::string>())(
      "print-egraph", "Print post-rewrite e-classes")(
      "allow-fp-reorder", "Allow floating-point operation reordering")(
      "allow-fp-reassociate", "Allow floating-point reassociation")(
      "allow-fp-distribute", "Allow floating-point distribution")(
      "assume-finite", "Assume floating-point values are finite")(
      "ignore-signed-zero", "Ignore signed zero")(
      "allow-dot-arithmetic", "Allow arithmetic rewrites of dot products");
}

bool applyRewriteCliOption(const std::string& name, const std::string& value,
                           TensorRewriteOptions& options) {
  if (name == "print-egraph") {
    options.print_egraph = parseBoolean(value);
    return true;
  }
  bool NumericalPermissions::* permission = nullptr;
  if (name == "allow-fp-reorder")
    permission = &NumericalPermissions::reorder_floating_point;
  else if (name == "allow-fp-reassociate")
    permission = &NumericalPermissions::reassociate_floating_point;
  else if (name == "allow-fp-distribute")
    permission = &NumericalPermissions::distribute_floating_point;
  else if (name == "assume-finite")
    permission = &NumericalPermissions::assume_finite;
  else if (name == "ignore-signed-zero")
    permission = &NumericalPermissions::ignore_signed_zero;
  else if (name == "allow-dot-arithmetic")
    permission = &NumericalPermissions::rewrite_dot_arithmetic;
  if (permission) {
    if (!options.semantic.permissions)
      options.semantic.permissions =
          numericalPermissions(options.numerical_policy);
    (*options.semantic.permissions).*permission = parseBoolean(value);
    return true;
  }
  if (name == "rules") {
    if (value.empty()) throw std::invalid_argument("empty rule path");
    options.rules_file = value;
  } else if (name == "numerical-policy") {
    if (value == "strict")
      options.numerical_policy = NumericalPolicy::PreserveEvaluation;
    else if (value == "relaxed")
      options.numerical_policy = NumericalPolicy::AllowReassociation;
    else
      throw std::invalid_argument("numerical policy must be strict or relaxed");
    options.semantic.permissions.reset();
  } else if (name == "rule-group") {
    if (value != "exact" && value != "algebra" && value != "all")
      throw std::invalid_argument("rule group must be exact, algebra, or all");
    options.semantic.enable_associativity = value != "exact";
    options.semantic.enable_linearity = value == "all";
    options.semantic.enable_homogeneity = value == "all";
  } else if (name == "iterations")
    options.runner.iteration_limit = positiveNumber(value);
  else if (name == "nodes")
    options.runner.node_limit = positiveNumber(value);
  else if (name == "matches") {
    options.runner.match_limit = positiveNumber(value);
    options.semantic.match_limit = positiveNumber(value);
  } else if (name == "search-visits")
    options.semantic.visit_limit = positiveNumber(value);
  else if (name == "time-ms")
    options.runner.time_limit = positiveMilliseconds(value);
  else if (name == "extractor") {
    if (value == "auto")
      options.extractor = TensorExtractorMode::Auto;
    else if (value == "tree")
      options.extractor = TensorExtractorMode::Tree;
    else
      throw std::invalid_argument("extractor must be auto or tree");
  } else if (name == "dag-states")
    options.dag.state_limit = positiveNumber(value);
  else if (name == "dag-time-ms")
    options.dag.time_limit = positiveMilliseconds(value);
  else if (name == "dag-frontier")
    options.dag.frontier_limit = positiveNumber(value);
  else if (name == "extraction") {
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

}  // namespace joint_shard

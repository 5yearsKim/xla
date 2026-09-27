#include "research/joint_shard/transforms/rewrite_options.h"

#include <charconv>
#include <limits>
#include <stdexcept>
namespace {
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

#include "research/joint_shard/tools/rewrite_cli_options.h"

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
                           TensorRewriteOptions& rewrite_options) {
  std::string argument = "--" + name;
  if (value == "true" &&
      (name == "print-egraph" || name == "allow-fp-reorder" ||
       name == "allow-fp-reassociate" || name == "allow-fp-distribute" ||
       name == "assume-finite" || name == "ignore-signed-zero" ||
       name == "allow-dot-arithmetic")) {
    // These switches are registered as Boolean cxxopts options.
  } else {
    argument += "=" + value;
  }
  return parseTensorRewriteOption(argument, rewrite_options);
}

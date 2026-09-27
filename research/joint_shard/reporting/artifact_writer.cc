#include "research/joint_shard/reporting/artifact_writer.h"

#include <stdexcept>

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "research/joint_shard/reporting/reports.h"
#include "research/joint_shard/search/region_optimizer.h"
#include "research/joint_shard/sharding/shardy_runner.h"

namespace joint_shard {
void ArtifactWriter::writeText(const std::string& relative,
                               const std::string& text) const {
  if (directory_.empty()) return;
  llvm::SmallString<256> path(directory_);
  llvm::sys::path::append(path, relative);
  if (auto error =
          llvm::sys::fs::create_directories(llvm::sys::path::parent_path(path)))
    throw std::runtime_error("cannot create artifact directory: " +
                             error.message());
  std::error_code error;
  llvm::raw_fd_ostream out(path, error);
  if (error)
    throw std::runtime_error("cannot write " + path.str().str() + ": " +
                             error.message());
  out << text;
  out.flush();
  if (out.has_error())
    throw std::runtime_error("failed to write " + path.str().str());
}
void ArtifactWriter::writeSnapshot(const ShardySnapshot& snapshot) const {
  writeText(snapshot.name + ".mlir", snapshot.mlir);
}
void ArtifactWriter::writeSelectedPrograms(const std::string& selected,
                                           const std::string& xla_input) const {
  if (directory_.empty())
    throw std::invalid_argument(
        "selected programs require an output directory");
  writeText("selected.mlir", selected);
  writeText("xla_input.mlir", xla_input);
}
OptimizationObserver ArtifactWriter::observer() const {
  if (directory_.empty()) return {};
  OptimizationObserver result;
  result.candidate_prepared = [this](size_t region, size_t candidate,
                                     mlir::ModuleOp module) {
    std::string text;
    llvm::raw_string_ostream out(text);
    module.print(out);
    out << '\n';
    writeText("region_" + std::to_string(region) + "/candidate_" +
                  std::to_string(candidate) + ".mlir",
              text);
  };
  result.snapshot = [this](size_t region, size_t boundary, size_t candidate,
                           const ShardySnapshot& snapshot) {
    writeText("region_" + std::to_string(region) + "/boundary_" +
                  std::to_string(boundary) + "/candidate_" +
                  std::to_string(candidate) + "/" + snapshot.name + ".mlir",
              snapshot.mlir);
  };
  result.region_completed = [this](const RegionSummary& region) {
    const auto directory = "region_" + std::to_string(region.id) + "/";
    writeText(directory + "summary.txt", formatReport(region));
    for (const auto& plan : region.plans)
      writeText(directory + "exact_P" + std::to_string(plan.id) + ".mlir",
                plan.lowered_mlir);
    for (size_t i = 0; i < region.frontier.size(); ++i)
      writeText(directory + "frontier_" + std::to_string(i) + ".mlir",
                region.plans.at(region.frontier[i]).lowered_mlir);
  };
  result.pair_completed = [this](const PairSummary& pair) {
    const auto directory = "pair_" + std::to_string(pair.a_region) + "_" +
                           std::to_string(pair.b_region) + "/";
    writeText(directory + "summary.txt", formatReport(pair));
    for (size_t i = 0; i < pair.plans.size(); ++i) {
      const auto prefix = directory + "selected_" + std::to_string(i);
      writeText(prefix + ".mlir", pair.plans[i].lowered_mlir);
      const auto& adapter = pair.plans[i].intermediate.lowered_mlir;
      writeText(prefix + "_adapter.mlir",
                adapter.empty()
                    ? "// Identity adapter: no operations, zero cost.\n"
                    : adapter);
    }
  };
  result.chain_completed = [this](const ChainExperiment& chain) {
    writeText("selected.mlir", chain.selected().lowered_mlir);
    writeText("selected_plan.txt", formatReport(chain));
    writeText("comparison.txt", formatComparison(chain));
  };
  result.dag_completed = [this](const DagExperiment& dag) {
    writeText("selected.mlir", dag.selected().lowered_mlir);
    writeText("selected_plan.txt", formatReport(dag));
    writeText("comparison.txt", formatComparison(dag));
  };
  return result;
}

}  // namespace joint_shard

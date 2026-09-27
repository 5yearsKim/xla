#ifndef RESEARCH_JOINT_SHARD_SHARDING_SHARDY_RUNNER_H_
#define RESEARCH_JOINT_SHARD_SHARDING_SHARDY_RUNNER_H_
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "research/joint_shard/sharding/module_statistics.h"

namespace joint_shard {
enum class ShardyStage { Propagation, ExplicitReshards, Collectives };
enum class SnapshotCapture { FinalOnly, AllStages };
struct ShardySnapshot {
  std::string name;
  std::string mlir;
  std::optional<ModuleStatistics> statistics;
};
struct ShardyRunOptions {
  ShardyStage stop_after = ShardyStage::Collectives;
  SnapshotCapture capture = SnapshotCapture::FinalOnly;
  bool collect_statistics = false;
  // Synchronous callback; snapshot is borrowed for the duration of the call.
  std::function<void(const ShardySnapshot&)> on_snapshot;
};
// Also used by isolated, already annotated reshard adapters.
mlir::LogicalResult lowerReshardsToCollectives(mlir::ModuleOp module);
// Mutates an annotated module; existing meshes and shardings are authoritative.
class ShardyRunner {
 public:
  mlir::LogicalResult run(mlir::ModuleOp module,
                          const ShardyRunOptions& options = {});
  const std::vector<ShardySnapshot>& snapshots() const { return snapshots_; }
  // Requires a successful run; transfers the final artifact without copying.
  std::string takeFinalMlir() { return std::move(snapshots_.back().mlir); }

 private:
  mlir::LogicalResult capture(mlir::ModuleOp module, const std::string& name,
                              const ShardyRunOptions& options, bool final);
  std::vector<ShardySnapshot> snapshots_;
};
}  // namespace joint_shard

#endif

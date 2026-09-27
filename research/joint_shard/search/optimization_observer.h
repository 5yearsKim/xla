#ifndef RESEARCH_JOINT_SHARD_SEARCH_OPTIMIZATION_OBSERVER_H_
#define RESEARCH_JOINT_SHARD_SEARCH_OPTIMIZATION_OBSERVER_H_
#include <cstddef>
#include <functional>

#include "mlir/IR/BuiltinOps.h"
namespace joint_shard {
struct ShardySnapshot;
struct RegionSummary;
struct PairSummary;
// Callbacks are synchronous; MLIR and summary references are borrowed for the
// call. Providing a snapshot callback enables all intermediate stages.
struct OptimizationObserver {
  std::function<void(size_t, size_t, mlir::ModuleOp)> candidate_prepared;
  std::function<void(size_t, size_t, size_t, const ShardySnapshot&)> snapshot;
  std::function<void(const RegionSummary&)> region_completed;
  std::function<void(const PairSummary&)> pair_completed;
};
}  // namespace joint_shard
#endif

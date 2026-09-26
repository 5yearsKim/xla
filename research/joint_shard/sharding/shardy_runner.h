#ifndef RESEARCH_JOINT_SHARD_SHARDING_SHARDY_RUNNER_H_
#define RESEARCH_JOINT_SHARD_SHARDING_SHARDY_RUNNER_H_

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

enum class ShardyStage { Propagation, ExplicitReshards, Collectives };

struct ShardyRunOptions {
  ShardyStage stopAfter = ShardyStage::Collectives;
  std::string dumpDirectory;
};

struct ModuleCost {
  // Logical tensor payload, a topology-independent proxy, not network bytes.
  uint64_t communication_payload_bytes = 0;
  uint64_t compute_work = 0;
  unsigned unknown_costs = 0;
};
ModuleCost estimateModuleCost(mlir::ModuleOp module);
// Conservative: an unknown estimate never displaces a measured baseline.
bool betterModuleCost(const ModuleCost& candidate, const ModuleCost& baseline);

struct ShardySnapshot {
  std::string name;
  std::string mlir;
  std::map<std::string, unsigned> communicationOps;
  ModuleCost cost;
};

// Mutates an annotated module; existing meshes and shardings are authoritative.
class ShardyRunner {
 public:
  mlir::LogicalResult run(mlir::ModuleOp module,
                          const ShardyRunOptions& options = {});
  const std::vector<ShardySnapshot>& snapshots() const { return snapshots_; }

 private:
  mlir::LogicalResult capture(mlir::ModuleOp module, const std::string& name,
                              const std::string& dumpDirectory);
  std::vector<ShardySnapshot> snapshots_;
};

#endif  // RESEARCH_JOINT_SHARD_SHARDING_SHARDY_RUNNER_H_

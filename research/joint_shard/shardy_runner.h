#ifndef RESEARCH_JOINT_SHARD_SHARDY_RUNNER_H_
#define RESEARCH_JOINT_SHARD_SHARDY_RUNNER_H_

#include <map>
#include <string>
#include <vector>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "research/joint_shard/boundary_sharding.h"

enum class ShardyStage { Propagation, ExplicitReshards, Collectives };

struct ShardyRunOptions {
  ShardyStage stopAfter = ShardyStage::Collectives;
  std::string dumpDirectory;
};

struct ShardySnapshot {
  std::string name;
  std::string mlir;
  std::map<std::string, unsigned> communicationOps;
};

// Mutates a fresh, unsharded module. Use separate clones for separate cases.
class ShardyRunner {
 public:
  mlir::LogicalResult run(mlir::ModuleOp module,
                          const std::vector<BoundarySharding>& constraints,
                          const ShardyRunOptions& options = {});
  const std::vector<ShardySnapshot>& snapshots() const { return snapshots_; }

 private:
  mlir::LogicalResult attachBoundaries(
      mlir::ModuleOp module, const std::vector<BoundarySharding>& constraints);
  mlir::LogicalResult capture(mlir::ModuleOp module, const std::string& name,
                              const std::string& dumpDirectory);
  std::vector<ShardySnapshot> snapshots_;
};

#endif  // RESEARCH_JOINT_SHARD_SHARDY_RUNNER_H_

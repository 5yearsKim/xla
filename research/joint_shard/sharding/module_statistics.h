#ifndef RESEARCH_JOINT_SHARD_SHARDING_MODULE_STATISTICS_H_
#define RESEARCH_JOINT_SHARD_SHARDING_MODULE_STATISTICS_H_
#include <cstdint>
#include <map>
#include <string>

#include "mlir/IR/BuiltinOps.h"

namespace joint_shard {
// Global logical work/payload diagnostics, independent of optimizer time costs.
struct ModuleStatistics {
  uint64_t communication_payload_bytes = 0;
  uint64_t compute_work = 0;
  unsigned unknown_costs = 0;
  std::map<std::string, unsigned> communication_ops;
};
ModuleStatistics collectModuleStatistics(mlir::ModuleOp module);
}  // namespace joint_shard

#endif

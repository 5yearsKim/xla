#ifndef RESEARCH_JOINT_SHARD_TRANSFORMS_REWRITE_REGIONS_H_
#define RESEARCH_JOINT_SHARD_TRANSFORMS_REWRITE_REGIONS_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

// Rewrite supported, unannotated computations between preserved operations.
// Sharded operations and explicit constraints remain in MLIR as boundaries;
// their results are opaque leaves and their operands are separate region roots.
mlir::LogicalResult rewriteUnconstrainedRegions(mlir::ModuleOp module);

#endif  // RESEARCH_JOINT_SHARD_TRANSFORMS_REWRITE_REGIONS_H_

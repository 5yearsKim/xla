#ifndef RESEARCH_JOINT_SHARD_EXPORT_SELECTED_EXPORT_H_
#define RESEARCH_JOINT_SHARD_EXPORT_SELECTED_EXPORT_H_

#include <cstdint>
#include <string>

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

namespace joint_shard {

struct ExportedSelectedProgram {
  // Global entry signature surrounding an already partitioned manual body,
  // exported by XLA's Shardy exporter for standard XLA compilation.
  mlir::OwningOpRef<mlir::ModuleOp> module;
  int64_t partitions = 1;
  std::string local_mlir;
};

// Consumes the post-propagation, explicit-collective output of the optimizer.
// Clones the input and uses upstream Shardy/XLA passes; does not propagate
// shardings or search again. The context must outlive the returned module.
absl::StatusOr<ExportedSelectedProgram> exportSelectedProgram(
    mlir::ModuleOp selected);

}  // namespace joint_shard

#endif  // RESEARCH_JOINT_SHARD_EXPORT_SELECTED_EXPORT_H_

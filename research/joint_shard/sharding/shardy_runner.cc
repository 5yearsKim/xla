#include "research/joint_shard/sharding/shardy_runner.h"

#include <optional>
#include <string>

#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/transforms/common/propagation_options.h"
#include "shardy/dialect/sdy/transforms/export/passes.h"
#include "shardy/dialect/sdy/transforms/propagation/passes.h"

namespace joint_shard {

mlir::LogicalResult ShardyRunner::capture(mlir::ModuleOp module,
                                          const std::string& name,
                                          const ShardyRunOptions& options,
                                          bool final) {
  if (mlir::failed(mlir::verify(module))) return mlir::failure();
  if (!final && options.capture == SnapshotCapture::FinalOnly)
    return mlir::success();
  ShardySnapshot snapshot;
  snapshot.name = name;
  if (options.collect_statistics)
    snapshot.statistics = collectModuleStatistics(module);
  llvm::raw_string_ostream out(snapshot.mlir);
  module.print(out);
  out << "\n";
  if (options.on_snapshot) options.on_snapshot(snapshot);
  snapshots_.push_back(std::move(snapshot));
  return mlir::success();
}

mlir::LogicalResult ShardyRunner::run(mlir::ModuleOp module,
                                      const ShardyRunOptions& options) {
  snapshots_.clear();
  if (mlir::failed(capture(module, "00_input", options, false))) {
    return mlir::failure();
  }
  std::optional<int64_t> partitionCount;
  for (auto mesh : module.getOps<mlir::sdy::MeshOp>()) {
    if (mesh.getMesh().empty() || mesh.getMesh().isMaximal()) continue;
    int64_t size = mesh.getMesh().getTotalSize();
    if (partitionCount && *partitionCount != size) {
      return mesh.emitError("meshes must have the same partition count");
    }
    partitionCount = size;
  }
  if (!partitionCount) {
    return module.emitError("expected an explicit non-empty partitioning mesh");
  }
  if (auto count =
          module->getAttrOfType<mlir::IntegerAttr>("mhlo.num_partitions")) {
    if (count.getInt() != *partitionCount) {
      return module.emitError("mhlo.num_partitions disagrees with mesh size");
    }
  }
  mlir::PassManager propagation(module.getContext());
  propagation.enableVerifier(true);
  mlir::sdy::PropagationOptions propagationOptions;
  propagationOptions.avoidExportForPartitioning = true;
  propagationOptions.keepShardingRules = true;
  propagationOptions.inlineMeshes = false;
  propagationOptions.updateNonDivisibleInputOutputShardings = false;
  propagationOptions.partitionCount = *partitionCount;
  mlir::sdy::addPropagationPipeline(propagation, propagationOptions);
  if (mlir::failed(propagation.run(module)) ||
      mlir::failed(capture(module, "01_propagated", options,
                           options.stop_after == ShardyStage::Propagation))) {
    return mlir::failure();
  }
  if (options.stop_after == ShardyStage::Propagation) return mlir::success();

  mlir::PassManager reshards(module.getContext());
  reshards.enableVerifier(true);
  mlir::sdy::InsertExplicitReshardsPassOptions reshardOptions;
  reshardOptions.enableFullVersion = true;
  // Propagation retains use-scoped constraints with avoidExportForPartitioning.
  reshards.addNestedPass<mlir::func::FuncOp>(
      mlir::sdy::createShardingConstraintToReshardPass());
  reshards.addNestedPass<mlir::func::FuncOp>(
      mlir::sdy::createInsertExplicitReshardsPass(reshardOptions));
  if (mlir::failed(reshards.run(module)) ||
      mlir::failed(
          capture(module, "02_explicit_reshards", options,
                  options.stop_after == ShardyStage::ExplicitReshards))) {
    return mlir::failure();
  }
  if (options.stop_after == ShardyStage::ExplicitReshards)
    return mlir::success();

  if (mlir::failed(lowerReshardsToCollectives(module))) return mlir::failure();
  return capture(module, "03_collectives", options, true);
}

mlir::LogicalResult lowerReshardsToCollectives(mlir::ModuleOp module) {
  mlir::PassManager collectives(module.getContext());
  collectives.enableVerifier(true);
  collectives.addNestedPass<mlir::func::FuncOp>(
      mlir::sdy::createReshardToCollectivesPass());
  return collectives.run(module);
}

}  // namespace joint_shard

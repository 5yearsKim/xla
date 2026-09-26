#include "research/joint_shard/sharding/shardy_runner.h"

#include <optional>
#include <string>

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/transforms/common/propagation_options.h"
#include "shardy/dialect/sdy/transforms/export/passes.h"
#include "shardy/dialect/sdy/transforms/propagation/passes.h"

mlir::LogicalResult ShardyRunner::capture(mlir::ModuleOp module,
                                          const std::string& name,
                                          const std::string& directory) {
  if (mlir::failed(mlir::verify(module))) return mlir::failure();
  ShardySnapshot snapshot;
  snapshot.name = name;
  llvm::raw_string_ostream out(snapshot.mlir);
  module.print(out);
  out << "\n";
  module.walk([&](mlir::Operation* op) {
    llvm::StringRef name = op->getName().getStringRef();
    if (name == "sdy.reshard" || name == "sdy.all_gather" ||
        name == "sdy.all_reduce" || name == "sdy.reduce_scatter" ||
        name == "sdy.all_to_all" || name == "sdy.collective_permute" ||
        name == "sdy.all_slice") {
      ++snapshot.communicationOps[name.str()];
    }
  });
  if (!directory.empty()) {
    if (auto error = llvm::sys::fs::create_directories(directory)) {
      return module.emitError("cannot create dump directory: ")
             << error.message();
    }
    llvm::SmallString<256> path(directory);
    llvm::sys::path::append(path, name + ".mlir");
    std::error_code error;
    llvm::raw_fd_ostream file(path, error);
    if (error)
      return module.emitError("cannot write snapshot: ") << error.message();
    file << snapshot.mlir;
    file.flush();
    if (file.has_error()) return module.emitError("failed to write snapshot");
  }
  snapshots_.push_back(std::move(snapshot));
  return mlir::success();
}

mlir::LogicalResult ShardyRunner::run(mlir::ModuleOp module,
                                      const ShardyRunOptions& options) {
  snapshots_.clear();
  if (mlir::failed(capture(module, "00_input", options.dumpDirectory))) {
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
      mlir::failed(capture(module, "01_propagated", options.dumpDirectory))) {
    return mlir::failure();
  }
  if (options.stopAfter == ShardyStage::Propagation) return mlir::success();

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
          capture(module, "02_explicit_reshards", options.dumpDirectory))) {
    return mlir::failure();
  }
  if (options.stopAfter == ShardyStage::ExplicitReshards)
    return mlir::success();

  mlir::PassManager collectives(module.getContext());
  collectives.enableVerifier(true);
  collectives.addNestedPass<mlir::func::FuncOp>(
      mlir::sdy::createReshardToCollectivesPass());
  if (mlir::failed(collectives.run(module))) return mlir::failure();
  return capture(module, "03_collectives", options.dumpDirectory);
}

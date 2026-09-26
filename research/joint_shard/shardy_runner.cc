#include "research/joint_shard/shardy_runner.h"

#include <set>
#include <string>
#include <tuple>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/transforms/common/propagation_options.h"
#include "shardy/dialect/sdy/transforms/export/passes.h"
#include "shardy/dialect/sdy/transforms/propagation/passes.h"

mlir::LogicalResult ShardyRunner::attachBoundaries(
    mlir::ModuleOp module, const std::vector<BoundarySharding>& constraints) {
  if (module.lookupSymbol("mesh")) {
    return module.emitError("expected an unsharded module without @mesh");
  }
  auto* context = module.getContext();
  if (!context->getOrLoadDialect("sdy")) {
    return module.emitError("Shardy dialect must be registered in the context");
  }
  auto mesh = mlir::sdy::MeshAttr::get(
      context, {mlir::sdy::MeshAxisAttr::get(context, "data", 2),
                mlir::sdy::MeshAxisAttr::get(context, "model", 2)});
  mlir::OpBuilder builder = mlir::OpBuilder::atBlockBegin(module.getBody());
  mlir::sdy::MeshOp::create(builder, module.getLoc(), "mesh", mesh);

  std::set<std::tuple<std::string, BoundarySharding::Kind, unsigned>> seen;
  for (const auto& boundary : constraints) {
    auto function =
        module.lookupSymbol<mlir::func::FuncOp>(boundary.functionName);
    if (!function || function.isExternal() ||
        !llvm::hasSingleElement(function.getBody())) {
      return module.emitError(
          "boundary requires a defined single-block function");
    }
    if (!seen.emplace(boundary.functionName, boundary.kind, boundary.index)
             .second) {
      return function.emitError("duplicate boundary sharding");
    }
    bool argument = boundary.kind == BoundarySharding::Kind::Argument;
    auto types =
        argument ? function.getArgumentTypes() : function.getResultTypes();
    if (boundary.index >= types.size()) {
      return function.emitError("boundary index is out of range");
    }
    auto type = llvm::dyn_cast<mlir::RankedTensorType>(types[boundary.index]);
    if (!type || !type.hasStaticShape() ||
        static_cast<std::size_t>(type.getRank()) !=
            boundary.sharding.dimensionAxes.size()) {
      return function.emitError(
          "boundary requires a static tensor with matching rank");
    }
    llvm::SmallVector<mlir::sdy::DimensionShardingAttr> dimensions;
    std::set<std::string> usedAxes;
    for (int64_t dim = 0; dim < type.getRank(); ++dim) {
      llvm::SmallVector<mlir::sdy::AxisRefAttr> axes;
      int64_t factor = 1;
      for (const std::string& axis : boundary.sharding.dimensionAxes[dim]) {
        if (axis != "data" && axis != "model") {
          return function.emitError(
                     "boundary references an unknown mesh axis: ")
                 << axis;
        }
        if (!usedAxes.insert(axis).second) {
          return function.emitError(
                     "mesh axis used more than once in a tensor: ")
                 << axis;
        }
        factor *= 2;
        axes.push_back(mlir::sdy::AxisRefAttr::get(context, axis));
      }
      if (type.getDimSize(dim) % factor != 0) {
        return function.emitError(
            "tensor dimension is not divisible by its sharding");
      }
      dimensions.push_back(mlir::sdy::DimensionShardingAttr::get(
          context, axes, /*isClosed=*/true));
    }
    auto sharding = mlir::sdy::TensorShardingAttr::get(
        context, "mesh", dimensions, /*replicatedAxes=*/{},
        /*unreducedAxes=*/{});
    if (argument) {
      function.setArgAttr(boundary.index, "sdy.sharding", sharding);
    } else {
      function.setResultAttr(boundary.index, "sdy.sharding", sharding);
    }
  }
  return mlir::success();
}

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

mlir::LogicalResult ShardyRunner::run(
    mlir::ModuleOp module, const std::vector<BoundarySharding>& constraints,
    const ShardyRunOptions& options) {
  snapshots_.clear();
  if (mlir::failed(capture(module, "00_exported", options.dumpDirectory)) ||
      mlir::failed(attachBoundaries(module, constraints)) ||
      mlir::failed(
          capture(module, "01_boundary_shardings", options.dumpDirectory))) {
    return mlir::failure();
  }
  mlir::PassManager propagation(module.getContext());
  propagation.enableVerifier(true);
  mlir::sdy::PropagationOptions propagationOptions;
  propagationOptions.avoidExportForPartitioning = true;
  propagationOptions.keepShardingRules = true;
  propagationOptions.inlineMeshes = false;
  propagationOptions.updateNonDivisibleInputOutputShardings = false;
  propagationOptions.partitionCount = 4;
  mlir::sdy::addPropagationPipeline(propagation, propagationOptions);
  if (mlir::failed(propagation.run(module)) ||
      mlir::failed(capture(module, "02_propagated", options.dumpDirectory))) {
    return mlir::failure();
  }
  if (options.stopAfter == ShardyStage::Propagation) return mlir::success();

  mlir::PassManager reshards(module.getContext());
  reshards.enableVerifier(true);
  mlir::sdy::InsertExplicitReshardsPassOptions reshardOptions;
  reshardOptions.enableFullVersion = true;
  reshards.addNestedPass<mlir::func::FuncOp>(
      mlir::sdy::createInsertExplicitReshardsPass(reshardOptions));
  if (mlir::failed(reshards.run(module)) ||
      mlir::failed(
          capture(module, "03_explicit_reshards", options.dumpDirectory))) {
    return mlir::failure();
  }
  if (options.stopAfter == ShardyStage::ExplicitReshards)
    return mlir::success();

  mlir::PassManager collectives(module.getContext());
  collectives.enableVerifier(true);
  collectives.addNestedPass<mlir::func::FuncOp>(
      mlir::sdy::createReshardToCollectivesPass());
  if (mlir::failed(collectives.run(module))) return mlir::failure();
  return capture(module, "04_collectives", options.dumpDirectory);
}

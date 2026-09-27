#include "research/joint_shard/sharding/shardy_runner.h"

#include <algorithm>
#include <limits>
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
#include "stablehlo/dialect/StablehloOps.h"

mlir::LogicalResult ShardyRunner::capture(mlir::ModuleOp module,
                                          const std::string& name,
                                          const std::string& directory) {
  if (mlir::failed(mlir::verify(module))) return mlir::failure();
  ShardySnapshot snapshot;
  snapshot.name = name;
  snapshot.cost = estimateModuleCost(module);
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

  if (mlir::failed(lowerReshardsToCollectives(module))) return mlir::failure();
  return capture(module, "03_collectives", options.dumpDirectory);
}

mlir::LogicalResult lowerReshardsToCollectives(mlir::ModuleOp module) {
  mlir::PassManager collectives(module.getContext());
  collectives.enableVerifier(true);
  collectives.addNestedPass<mlir::func::FuncOp>(
      mlir::sdy::createReshardToCollectivesPass());
  return collectives.run(module);
}

namespace {
uint64_t elementBits(mlir::Type type) {
  if (auto i = llvm::dyn_cast<mlir::IntegerType>(type)) return i.getWidth();
  if (auto f = llvm::dyn_cast<mlir::FloatType>(type)) return f.getWidth();
  if (auto c = llvm::dyn_cast<mlir::ComplexType>(type))
    return 2 * elementBits(c.getElementType());
  return 0;
}
std::optional<uint64_t> elements(mlir::Type type) {
  auto tensor = llvm::dyn_cast<mlir::RankedTensorType>(type);
  if (!tensor || !tensor.hasStaticShape()) return {};
  uint64_t total = 1;
  for (auto dimension : tensor.getShape()) {
    if (dimension && total > std::numeric_limits<uint64_t>::max() / dimension)
      return {};
    total *= dimension;
  }
  return total;
}
std::optional<uint64_t> payload(mlir::Type type) {
  auto count = elements(type);
  auto tensor = llvm::dyn_cast<mlir::RankedTensorType>(type);
  if (!count || !tensor) return {};
  auto bits = elementBits(tensor.getElementType());
  if (!bits || *count > (std::numeric_limits<uint64_t>::max() - 7) / bits)
    return {};
  return (*count * bits + 7) / 8;
}
void addCost(uint64_t& into, std::optional<uint64_t> value, unsigned& unknown) {
  if (!value || *value > std::numeric_limits<uint64_t>::max() - into) {
    ++unknown;
    return;
  }
  into += *value;
}
}  // namespace
ModuleCost estimateModuleCost(mlir::ModuleOp module) {
  ModuleCost cost;
  module.walk([&](mlir::Operation* op) {
    auto name = op->getName().getStringRef();
    if (name == "sdy.reshard" || name == "sdy.all_gather" ||
        name == "sdy.all_reduce" || name == "sdy.reduce_scatter" ||
        name == "sdy.all_to_all" || name == "sdy.collective_permute") {
      // Use the largest logical operand/result payload once per collective.
      // Device groups, local partition sizes, and topology are not modeled.
      uint64_t largest = 0;
      bool known = true;
      for (auto type : op->getOperandTypes()) {
        auto bytes = payload(type);
        if (!bytes)
          known = false;
        else
          largest = std::max(largest, *bytes);
      }
      for (auto type : op->getResultTypes()) {
        auto bytes = payload(type);
        if (!bytes)
          known = false;
        else
          largest = std::max(largest, *bytes);
      }
      addCost(cost.communication_payload_bytes,
              known ? std::optional<uint64_t>(largest) : std::nullopt,
              cost.unknown_costs);
    }
    if (!name.starts_with("stablehlo.") || op->getNumResults() != 1 ||
        name == "stablehlo.constant" || name == "stablehlo.reshape" ||
        name == "stablehlo.broadcast_in_dim")
      return;
    auto work = elements(op->getResult(0).getType());
    if (name == "stablehlo.dot_general" && work) {
      auto dims = op->getAttrOfType<mlir::stablehlo::DotDimensionNumbersAttr>(
          "dot_dimension_numbers");
      auto lhs =
          llvm::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
      if (!dims || !lhs || !lhs.hasStaticShape())
        work.reset();
      else {
        for (auto axis : dims.getLhsContractingDimensions()) {
          auto dimension = static_cast<uint64_t>(lhs.getDimSize(axis));
          if (dimension &&
              *work > std::numeric_limits<uint64_t>::max() / dimension) {
            work.reset();
            break;
          }
          *work *= dimension;
        }
        if (work && *work <= std::numeric_limits<uint64_t>::max() / 2)
          *work *= 2;
        else
          work.reset();
      }
    } else if (name == "stablehlo.reduce") {
      work = elements(op->getOperand(0).getType());
    } else if (name == "stablehlo.exponential" || name == "stablehlo.log" ||
               name == "stablehlo.sqrt" || name == "stablehlo.tanh") {
      if (work && *work <= std::numeric_limits<uint64_t>::max() / 8)
        *work *= 8;
      else
        work.reset();
    }
    addCost(cost.compute_work, work, cost.unknown_costs);
  });
  return cost;
}

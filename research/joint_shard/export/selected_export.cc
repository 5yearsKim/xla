#include "research/joint_shard/export/selected_export.h"

#include <utility>

#include "absl/status/status.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/export/passes.h"
#include "xla/service/spmd/shardy/stablehlo_round_trip/stablehlo_export.h"

namespace joint_shard {
namespace {

absl::Status invalid(const char* message) {
  return absl::InvalidArgumentError(message);
}

absl::Status validateLayout(mlir::sdy::TensorShardingAttr sharding,
                            mlir::Type type, mlir::sdy::MeshOp mesh) {
  auto tensor = llvm::dyn_cast<mlir::RankedTensorType>(type);
  if (!tensor || !tensor.hasStaticShape())
    return invalid("selected program requires static ranked tensors");
  if (!sharding || sharding.getMesh(mesh) != mesh.getMesh())
    return invalid("selected program requires shardings on its single mesh");
  if (sharding.getDimShardings().size() != tensor.getRank())
    return invalid("sharding rank does not match tensor rank");
  for (auto [size, dim] :
       llvm::zip(tensor.getShape(), sharding.getDimShardings())) {
    if (!dim.getIsClosed())
      return invalid("selected program requires closed shardings");
    int64_t shards = 1;
    for (auto axis : dim.getAxes()) shards *= axis.getSize(mesh.getMesh());
    if (size % shards != 0)
      return invalid("selected program requires divisible shardings");
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<ExportedSelectedProgram> exportSelectedProgram(
    mlir::ModuleOp selected) {
  if (mlir::failed(mlir::verify(selected)))
    return invalid("selected module failed MLIR verification");
  mlir::sdy::MeshOp mesh;
  mlir::func::FuncOp entry;
  for (auto& op : selected.getBody()->getOperations()) {
    if (auto candidate = llvm::dyn_cast<mlir::sdy::MeshOp>(op)) {
      if (mesh) return invalid("expected exactly one mesh");
      mesh = candidate;
    } else if (auto candidate = llvm::dyn_cast<mlir::func::FuncOp>(op)) {
      if (entry || candidate.getSymName() != "main")
        return invalid("expected exactly one function named main");
      entry = candidate;
    } else {
      return invalid("unsupported selected module top-level operation");
    }
  }
  if (!mesh || mesh.getMesh().empty() || mesh.getMesh().isMaximal() || !entry ||
      entry.isExternal() || !entry.getBody().hasOneBlock())
    return invalid("expected one partitioning mesh and a single-block main");
  const int64_t partitions = mesh.getMesh().getTotalSize();
  if (auto count =
          selected->getAttrOfType<mlir::IntegerAttr>("mhlo.num_partitions");
      count && count.getInt() != partitions)
    return invalid("mhlo.num_partitions disagrees with mesh size");
  if (auto count =
          selected->getAttrOfType<mlir::IntegerAttr>("mhlo.num_replicas");
      count && count.getInt() != 1)
    return invalid("selected program requires one XLA replica");

  llvm::SmallVector<mlir::sdy::TensorShardingAttr> inputs, outputs;
  for (unsigned i = 0; i < entry.getNumArguments(); ++i) {
    auto layout = entry.getArgAttrOfType<mlir::sdy::TensorShardingAttr>(
        i, "sdy.sharding");
    auto status = validateLayout(layout, entry.getArgument(i).getType(), mesh);
    if (!status.ok()) return status;
    if (!layout.getUnreducedAxes().empty())
      return invalid("unreduced entry arguments are unsupported");
    inputs.push_back(layout);
  }
  for (unsigned i = 0; i < entry.getNumResults(); ++i) {
    auto layout = entry.getResultAttrOfType<mlir::sdy::TensorShardingAttr>(
        i, "sdy.sharding");
    auto status = validateLayout(layout, entry.getResultTypes()[i], mesh);
    if (!status.ok()) return status;
    if (!layout.getUnreducedAxes().empty())
      return invalid("unreduced entry results are unsupported");
    outputs.push_back(layout);
  }
  absl::Status status = absl::OkStatus();
  entry.walk([&](mlir::Operation* op) {
    if (!status.ok() || op == entry.getOperation()) return;
    auto name = op->getName().getStringRef();
    if (name == "sdy.reshard" || name == "sdy.sharding_constraint" ||
        name == "sdy.manual_computation" || name == "sdy.named_computation") {
      status = invalid(
          "expected global post-propagation collective IR; "
          "unresolved reshards, constraints, or manual bodies remain");
      return;
    }
    // Reducer block arguments are scalars local to the reduction, not ports.
    for (auto value : op->getResults()) {
      auto tensor = llvm::dyn_cast<mlir::RankedTensorType>(value.getType());
      if (!tensor || !tensor.hasStaticShape()) {
        status = invalid("selected program requires static ranked tensors");
        return;
      }
      auto layout = mlir::sdy::getSharding(value);
      if (layout) status = validateLayout(layout, value.getType(), mesh);
      if (!status.ok()) return;
    }
  });
  if (!status.ok()) return status;

  auto* context = selected.getContext();
  mlir::OwningOpRef<mlir::ModuleOp> local(selected.clone());
  mlir::PassManager partitioner(context);
  partitioner.enableVerifier(true);
  // The optimizer retains propagation rules with global factor sizes. They
  // are no longer needed and would be stale after changing tensor shapes.
  partitioner.addNestedPass<mlir::func::FuncOp>(
      mlir::sdy::createDropShardingRulesPass());
  mlir::sdy::ConvertGlobalToLocalPassOptions conversion;
  conversion.replicaCount = 1;
  conversion.partitionCount = partitions;
  // Dense replica groups are supported by both CPU and GPU XLA backends.
  conversion.enableRGV3 = false;
  partitioner.addPass(mlir::sdy::createConvertGlobalToLocalPass(conversion));
  partitioner.addPass(mlir::sdy::createDropShardingAndMeshPass());
  if (mlir::failed(partitioner.run(*local)))
    return invalid("Shardy global-to-local lowering failed");

  ExportedSelectedProgram result;
  result.partitions = partitions;
  llvm::raw_string_ostream local_out(result.local_mlir);
  local->print(local_out);

  // Follow upstream per-instruction partitioning's integration pattern:
  // retain the global entry contract and enclose local code in a manual body.
  result.module = mlir::ModuleOp::create(selected.getLoc());
  (*result.module)->setAttrs(selected->getAttrs());
  mlir::OpBuilder builder(context);
  (*result.module)
      ->setAttr("mhlo.num_partitions", builder.getI64IntegerAttr(partitions));
  (*result.module)->setAttr("mhlo.num_replicas", builder.getI64IntegerAttr(1));
  builder.setInsertionPointToStart(result.module->getBody());
  builder.clone(*mesh.getOperation());
  auto wrapper = mlir::func::FuncOp::create(builder, entry.getLoc(), "main",
                                            entry.getFunctionType());
  wrapper.setAllArgAttrs(entry.getAllArgAttrs());
  wrapper.setAllResultAttrs(entry.getAllResultAttrs());
  wrapper.addEntryBlock();
  builder.setInsertionPointToStart(&wrapper.getBody().front());
  llvm::SmallVector<mlir::StringAttr> axes;
  for (auto axis : mesh.getMesh().getAxes())
    axes.push_back(builder.getStringAttr(axis.getName()));
  auto manual = mlir::sdy::ManualComputationOp::create(
      builder, entry.getLoc(), entry.getResultTypes(), wrapper.getArguments(),
      inputs, outputs, axes);
  auto local_entry = local->lookupSymbol<mlir::func::FuncOp>("main");
  manual.getBody().takeBody(local_entry.getBody());
  auto* terminator = manual.getBody().front().getTerminator();
  builder.setInsertionPoint(terminator);
  mlir::sdy::ReturnOp::create(builder, terminator->getLoc(),
                              terminator->getOperands());
  terminator->erase();
  builder.setInsertionPointAfter(manual);
  mlir::func::ReturnOp::create(builder, entry.getLoc(), manual.getResults());
  if (mlir::failed(mlir::verify(*result.module)))
    return invalid("manual wrapper failed verification");

  mlir::PassManager exporter(context);
  exporter.enableVerifier(true);
  xla::sdy::StablehloExportPipelineOptions export_options;
  // PJRT imports StableHLO, not MHLO copy operations. Keep boundary layout
  // anchors as supported @Sharding custom calls using the upstream option.
  export_options.keepHloShardingConstraints = true;
  xla::sdy::addStablehloExportPipeline(exporter, export_options);
  if (mlir::failed(exporter.run(*result.module)))
    return invalid("XLA Shardy export failed");
  return result;
}

}  // namespace joint_shard

#include "research/joint_shard/sharding/cost_model.h"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace joint_shard {

namespace {
std::optional<double> elements(mlir::Type type) {
  auto tensor = llvm::dyn_cast<mlir::RankedTensorType>(type);
  if (!tensor || !tensor.hasStaticShape()) return {};
  double count = 1;
  for (auto size : tensor.getShape()) count *= size;
  return std::isfinite(count) ? std::optional<double>(count) : std::nullopt;
}
std::optional<double> partitionFactor(mlir::Value value,
                                      mlir::sdy::MeshAttr mesh) {
  auto sharding = mlir::sdy::getSharding(value);
  if (!sharding) return 1;
  double factor = 1;
  for (auto dimension : sharding.getDimShardings())
    for (auto axis : dimension.getAxes()) {
      if (!mesh.hasAxis(axis.getName())) return {};
      factor *= axis.getSize(mesh);
    }
  if (!std::isfinite(factor) || factor <= 0) return {};
  return factor;
}
std::optional<double> localElements(mlir::Value value,
                                    mlir::sdy::MeshAttr mesh) {
  auto count = elements(value.getType());
  auto factor = partitionFactor(value, mesh);
  if (!count || !factor) return {};
  return *count / *factor;
}
std::optional<double> localBytes(mlir::Value value, mlir::sdy::MeshAttr mesh) {
  auto count = localElements(value, mesh);
  auto tensor = llvm::dyn_cast<mlir::RankedTensorType>(value.getType());
  if (!count || !tensor) return {};
  auto type = tensor.getElementType();
  unsigned bits = 0;
  if (auto integer = llvm::dyn_cast<mlir::IntegerType>(type))
    bits = integer.getWidth();
  else if (auto floating = llvm::dyn_cast<mlir::FloatType>(type))
    bits = floating.getWidth();
  else if (auto complex = llvm::dyn_cast<mlir::ComplexType>(type)) {
    if (auto floating =
            llvm::dyn_cast<mlir::FloatType>(complex.getElementType()))
      bits = 2 * floating.getWidth();
  }
  if (!bits) return {};
  return std::ceil(*count * bits / 8);
}
double groupSize(llvm::ArrayRef<mlir::sdy::AxisRefAttr> axes,
                 mlir::sdy::MeshAttr mesh) {
  double size = 1;
  for (auto axis : axes) {
    if (!mesh.hasAxis(axis.getName())) return 0;
    size *= axis.getSize(mesh);
  }
  return size;
}
}  // namespace

CostModel::CostModel(CostModelOptions options) : options_(options) {
  if (!std::isfinite(options.compute_work_per_us) ||
      options.compute_work_per_us <= 0 ||
      !std::isfinite(options.bandwidth_bytes_per_us) ||
      options.bandwidth_bytes_per_us <= 0 ||
      !std::isfinite(options.collective_latency_us) ||
      options.collective_latency_us < 0)
    throw std::invalid_argument(
        "cost rates must be positive and latency nonnegative");
}

Cost CostModel::estimate(mlir::ModuleOp module,
                         mlir::sdy::MeshAttr mesh) const {
  Cost result;
  for (auto function : module.getOps<mlir::func::FuncOp>()) {
    if (function.isExternal()) {
      ++result.unknown;
      continue;
    }
    // Reducer scalar bodies are part of the parent reduction's work.
    for (auto& block : function.getBody())
      for (auto& op : block) {
        auto name = op.getName().getStringRef();
        if (name == "func.return" || name == "stablehlo.constant" ||
            name == "sdy.constant" || name == "stablehlo.reshape" ||
            name == "stablehlo.broadcast_in_dim" || name == "sdy.all_slice")
          continue;
        if (auto collective =
                llvm::dyn_cast<mlir::sdy::CollectiveOpInterface>(&op)) {
          auto input = localBytes(op.getOperand(0), mesh);
          auto output = localBytes(op.getResult(0), mesh);
          if (!input || !output) {
            ++result.unknown;
            continue;
          }
          double bytes = 0;
          if (name == "sdy.all_gather")
            bytes = std::max(0.0, *output - *input);
          else if (name == "sdy.all_reduce") {
            double group = groupSize(
                llvm::cast<mlir::sdy::AllReduceOp>(&op).getReductionAxes(),
                mesh);
            if (group < 1) {
              ++result.unknown;
              continue;
            }
            bytes = 2 * (group - 1) / group * *input;
          } else if (name == "sdy.reduce_scatter")
            bytes = std::max(0.0, *input - *output);
          else if (name == "sdy.all_to_all") {
            double group = 1;
            for (auto parameter :
                 llvm::cast<mlir::sdy::AllToAllOp>(&op).getParams())
              group *= groupSize(parameter.getAxes(), mesh);
            if (group < 1) {
              ++result.unknown;
              continue;
            }
            bytes = (group - 1) / group * *input;
          } else if (name == "sdy.collective_permute")
            bytes = *input;
          else {
            ++result.unknown;
            continue;
          }
          result.communication += options_.collective_latency_us +
                                  bytes / options_.bandwidth_bytes_per_us;
          continue;
        }
        if (!name.starts_with("stablehlo.") || op.getNumResults() != 1) {
          ++result.unknown;
          continue;
        }
        auto work = localElements(op.getResult(0), mesh);
        if (auto dot = llvm::dyn_cast<mlir::stablehlo::DotGeneralOp>(&op)) {
          auto lhs = llvm::cast<mlir::RankedTensorType>(dot.getLhs().getType());
          auto sharding = mlir::sdy::getSharding(dot.getLhs());
          if (!work || !lhs.hasStaticShape())
            work.reset();
          else {
            for (auto dim :
                 dot.getDotDimensionNumbers().getLhsContractingDimensions()) {
              double local = lhs.getDimSize(dim);
              if (sharding)
                for (auto axis : sharding.getDimSharding(dim).getAxes())
                  local /= axis.getSize(mesh);
              *work *= local;
            }
            *work *= 2;
          }
        } else if (name == "stablehlo.reduce")
          work = localElements(op.getOperand(0), mesh);
        else if (name == "stablehlo.exponential" || name == "stablehlo.log" ||
                 name == "stablehlo.sqrt" || name == "stablehlo.tanh") {
          if (work) *work *= 8;
        }
        if (!work || !std::isfinite(*work))
          ++result.unknown;
        else
          result.compute += *work / options_.compute_work_per_us;
      }
  }
  return result;
}

}  // namespace joint_shard

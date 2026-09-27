#include "research/joint_shard/sharding/module_statistics.h"

#include <algorithm>
#include <limits>
#include <optional>

#include "stablehlo/dialect/StablehloOps.h"

namespace joint_shard {
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
void addCounter(uint64_t& into, std::optional<uint64_t> value,
                unsigned& unknown) {
  if (!value || *value > std::numeric_limits<uint64_t>::max() - into) {
    ++unknown;
    return;
  }
  into += *value;
}
}  // namespace
ModuleStatistics collectModuleStatistics(mlir::ModuleOp module) {
  ModuleStatistics statistics;
  module.walk([&](mlir::Operation* op) {
    auto name = op->getName().getStringRef();
    if (name == "sdy.reshard" || name == "sdy.all_gather" ||
        name == "sdy.all_reduce" || name == "sdy.reduce_scatter" ||
        name == "sdy.all_to_all" || name == "sdy.collective_permute" ||
        name == "sdy.all_slice")
      ++statistics.communication_ops[name.str()];

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
      addCounter(statistics.communication_payload_bytes,
                 known ? std::optional<uint64_t>(largest) : std::nullopt,
                 statistics.unknown_costs);
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
    addCounter(statistics.compute_work, work, statistics.unknown_costs);
  });
  return statistics;
}

}  // namespace joint_shard

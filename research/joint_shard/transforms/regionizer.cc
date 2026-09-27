#include "research/joint_shard/transforms/regionizer.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <stdexcept>
#include <tuple>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "research/joint_shard/bridge/stablehlo_importer.h"

namespace joint_shard {

Region describeRegion(llvm::ArrayRef<mlir::Operation*> operations) {
  Region region;
  region.operations.assign(operations.begin(), operations.end());
  llvm::DenseSet<mlir::Operation*> members(operations.begin(),
                                           operations.end());
  llvm::DenseSet<mlir::Value> inputs;
  for (auto* op : operations) {
    for (auto value : op->getOperands())
      if (!members.contains(value.getDefiningOp()) &&
          inputs.insert(value).second)
        region.inputs.push_back(value);
    for (auto value : op->getResults())
      if (llvm::any_of(value.getUses(), [&](mlir::OpOperand& use) {
            return !members.contains(use.getOwner());
          }))
        region.outputs.push_back(value);
  }
  return region;
}

size_t countCrossingValues(llvm::ArrayRef<mlir::Operation*> operations,
                           size_t cut) {
  if (!cut || cut >= operations.size())
    throw std::invalid_argument("cut must be inside an operation range");
  llvm::DenseSet<mlir::Operation*> before(operations.begin(),
                                          operations.begin() + cut);
  size_t count = 0;
  for (auto* op : operations.take_front(cut))
    for (auto value : op->getResults())
      if (llvm::any_of(value.getUses(), [&](mlir::OpOperand& use) {
            return !before.contains(use.getOwner());
          }))
        ++count;
  return count;
}

bool cutBreaksProtectedPattern(llvm::ArrayRef<mlir::Operation*> operations,
                               size_t cut) {
  llvm::DenseMap<mlir::Operation*, size_t> positions;
  for (size_t i = 0; i < operations.size(); ++i) positions[operations[i]] = i;
  auto crosses = [&](mlir::Operation* producer, size_t consumer) {
    auto found = positions.find(producer);
    return found != positions.end() && found->second < cut && consumer >= cut;
  };
  for (size_t i = 0; i < operations.size(); ++i) {
    auto* op = operations[i];
    auto name = op->getName().getStringRef();
    for (auto operand : op->getOperands()) {
      auto* producer = operand.getDefiningOp();
      if (!producer) continue;
      auto producerName = producer->getName().getStringRef();
      if ((name == "stablehlo.dot_general" &&
           producerName == "stablehlo.transpose") ||
          (name == "stablehlo.reshape" && producerName == "stablehlo.reshape"))
        if (crosses(producer, i)) return true;
      if (name == "stablehlo.dot_general" &&
          producerName == "stablehlo.multiply") {
        for (auto input : producer->getOperands()) {
          auto* broadcast = input.getDefiningOp();
          if (broadcast && positions.contains(producer) &&
              broadcast->getName().getStringRef() ==
                  "stablehlo.broadcast_in_dim" &&
              positions.contains(broadcast) && crosses(broadcast, i))
            return true;
        }
      }
    }
  }
  return false;
}

std::vector<Region> Regionizer::split(mlir::func::FuncOp function) const {
  if (!options_.max_region_ops)
    throw std::invalid_argument("max_region_ops must be positive");
  if (function.isExternal()) return {};
  if (!llvm::hasSingleElement(function.getBody()))
    throw std::invalid_argument(
        "region search requires single-block functions");
  std::vector<Region> regions;
  std::function<void(llvm::ArrayRef<mlir::Operation*>)> splitRange =
      [&](llvm::ArrayRef<mlir::Operation*> range) {
        if (range.empty()) return;
        size_t best = 0;
        auto score = std::make_tuple(std::numeric_limits<size_t>::max(),
                                     std::numeric_limits<size_t>::max());
        if (range.size() > options_.max_region_ops) {
          for (size_t cut = 1; cut < range.size(); ++cut) {
            if (cutBreaksProtectedPattern(range, cut)) continue;
            size_t distance = cut * 2 > range.size() ? cut * 2 - range.size()
                                                     : range.size() - cut * 2;
            auto candidate =
                std::make_tuple(countCrossingValues(range, cut), distance);
            if (candidate < score) {
              score = candidate;
              best = cut;
            }
          }
        }
        if (best) {
          splitRange(range.take_front(best));
          splitRange(range.drop_front(best));
          return;
        }
        auto region = describeRegion(range);
        region.id = regions.size();
        region.oversized = range.size() > options_.max_region_ops;
        regions.push_back(std::move(region));
      };
  std::vector<mlir::Operation*> island;
  for (auto& op : function.getBody().front()) {
    if (!canImportTensorOperation(&op)) {
      splitRange(island);
      island.clear();
    } else if (std::find(options_.rule_blockers.begin(),
                         options_.rule_blockers.end(),
                         op.getName().getStringRef().str()) !=
               options_.rule_blockers.end()) {
      splitRange(island);
      island.clear();
      splitRange({&op});
    } else {
      island.push_back(&op);
    }
  }
  splitRange(island);
  return regions;
}

}  // namespace joint_shard

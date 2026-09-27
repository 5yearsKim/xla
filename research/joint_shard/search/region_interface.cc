#include "research/joint_shard/search/region_interface.h"

#include <algorithm>
#include <set>
#include <stdexcept>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "research/joint_shard/transforms/regionizer.h"

namespace joint_shard {
ValueIndex::ValueIndex(mlir::ModuleOp module) {
  size_t next = 0;
  for (auto function : module.getOps<mlir::func::FuncOp>()) {
    for (auto& block : function.getBody()) {
      for (auto arg : block.getArguments())
        ids_.emplace(arg.getAsOpaquePointer(), next++);
      for (auto& op : block)
        for (auto result : op.getResults())
          ids_.emplace(result.getAsOpaquePointer(), next++);
    }
  }
}
RegionInterface ValueIndex::interface(const Region& region) const {
  RegionInterface result;
  for (auto value : region.inputs)
    result.inputs.push_back(
        {ids_.at(value.getAsOpaquePointer()), value.getType()});
  for (auto value : region.outputs)
    result.outputs.push_back(
        {ids_.at(value.getAsOpaquePointer()), value.getType()});
  return result;
}
TensorPort ValueIndex::port(mlir::Value value) const {
  return {ids_.at(value.getAsOpaquePointer()), value.getType()};
}
namespace {
std::map<ValueId, ValueLifetime> lifetimes(
    const RegionInterface& external,
    const std::vector<RegionInterface>& regions) {
  std::map<ValueId, ValueLifetime> result;
  for (auto port : external.inputs)
    if (!result.emplace(port.value, ValueLifetime{port, {}, {}}).second)
      throw std::invalid_argument("duplicate DAG function argument ID");
  auto consume = [&](TensorPort port, size_t consumer) {
    auto found = result.find(port.value);
    if (found == result.end() || found->second.port != port)
      throw std::invalid_argument("DAG value has no earlier typed producer");
    auto& uses = found->second.consumers;
    if (uses.empty() || uses.back() != consumer) uses.push_back(consumer);
  };
  for (size_t i = 0; i < regions.size(); ++i) {
    std::set<ValueId> inputs;
    for (auto port : regions[i].inputs) {
      if (!inputs.insert(port.value).second)
        throw std::invalid_argument(
            "DAG region input ports must be deduplicated");
      consume(port, i);
    }
    for (auto port : regions[i].outputs)
      if (!result.emplace(port.value, ValueLifetime{port, i, {}}).second)
        throw std::invalid_argument("DAG value has multiple producers");
  }
  for (auto port : external.outputs) consume(port, regions.size());
  return result;
}
std::vector<std::vector<TensorPort>> cuts(
    const std::map<ValueId, ValueLifetime>& values, size_t count) {
  std::vector<std::vector<TensorPort>> result(count);
  for (const auto& [id, value] : values) {
    if (!value.producer || value.consumers.empty()) continue;
    for (size_t i = *value.producer; i < value.consumers.back(); ++i)
      result[i].push_back(value.port);
  }
  return result;
}
}  // namespace
std::vector<std::vector<TensorPort>> buildLiveCuts(
    const RegionInterface& external,
    const std::vector<RegionInterface>& regions) {
  return cuts(lifetimes(external, regions), regions.size());
}
DagInterface buildDagInterface(mlir::func::FuncOp function,
                               const std::vector<Region>& regions,
                               const ValueIndex& values) {
  if (function.isExternal() || !function.getBody().hasOneBlock() ||
      regions.empty())
    throw std::invalid_argument(
        "DAG requires a nonempty single-block function");
  auto& block = function.getBody().front();
  auto ret = llvm::dyn_cast<mlir::func::ReturnOp>(block.getTerminator());
  if (!ret || !ret.getNumOperands())
    throw std::invalid_argument("DAG requires tensor function results");
  DagInterface result;
  for (auto arg : function.getArguments())
    result.external.inputs.push_back(values.port(arg));
  for (auto value : ret.getOperands())
    result.external.outputs.push_back(values.port(value));
  auto* next = &block.front();
  for (const auto& region : regions) {
    if (region.operations.empty())
      throw std::invalid_argument("empty DAG region");
    for (auto* op : region.operations) {
      if (op != next)
        throw std::invalid_argument(
            "DAG does not cover all function operations (preserved boundary)");
      next = op->getNextNode();
    }
    result.regions.push_back(values.interface(region));
  }
  if (next != ret.getOperation())
    throw std::invalid_argument("DAG does not cover all function operations");
  result.values = lifetimes(result.external, result.regions);
  result.live_after = cuts(result.values, regions.size());
  return result;
}
ChainInterface buildChainInterface(mlir::func::FuncOp function,
                                   const std::vector<Region>& regions,
                                   const ValueIndex& values) {
  if (function.isExternal() || !function.getBody().hasOneBlock() ||
      regions.empty())
    throw std::invalid_argument(
        "chain requires a nonempty single-block function");
  auto& block = function.getBody().front();
  auto ret = llvm::dyn_cast<mlir::func::ReturnOp>(block.getTerminator());
  if (!ret || !ret.getNumOperands())
    throw std::invalid_argument("chain requires tensor function results");
  ChainInterface result;
  for (auto arg : function.getArguments())
    result.external.inputs.push_back(values.port(arg));
  for (auto value : ret.getOperands())
    result.external.outputs.push_back(values.port(value));
  auto* next = &block.front();
  for (size_t i = 0; i < regions.size(); ++i) {
    const auto& region = regions[i];
    if (region.operations.empty())
      throw std::invalid_argument("empty chain region");
    for (auto* op : region.operations) {
      if (op != next)
        throw std::invalid_argument(
            "chain does not cover all function operations (preserved "
            "boundary)");
      next = op->getNextNode();
    }
    if (i + 1 < regions.size())
      (void)buildPairInterface(region, regions[i + 1], values);
    result.regions.push_back(values.interface(region));
    std::vector<int64_t> bindings;
    size_t intermediate = 0;
    for (size_t j = 0; j < region.inputs.size(); ++j) {
      auto value = region.inputs[j];
      if (i && value == regions[i - 1].outputs[0]) {
        bindings.push_back(-1);
        intermediate = j;
      } else {
        auto arg = llvm::dyn_cast<mlir::BlockArgument>(value);
        if (!arg || arg.getOwner() != &block)
          throw std::invalid_argument(
              "chain input must be a function argument or preceding output");
        bindings.push_back(arg.getArgNumber());
      }
    }
    result.inputs.push_back(std::move(bindings));
    result.intermediate_inputs.push_back(intermediate);
  }
  if (next != ret.getOperation())
    throw std::invalid_argument("chain does not cover all function operations");
  for (auto port : result.external.outputs)
    if (std::find(result.regions.back().outputs.begin(),
                  result.regions.back().outputs.end(),
                  port) == result.regions.back().outputs.end())
      throw std::invalid_argument("chain results must be final region outputs");
  return result;
}
PairInterface buildPairInterface(const Region& a, const Region& b,
                                 const ValueIndex& values) {
  if (a.operations.empty() || b.operations.empty() || a.outputs.size() != 1 ||
      a.operations.back()->getNextNode() != b.operations.front() ||
      a.operations.front()->getBlock() != b.operations.front()->getBlock())
    throw std::invalid_argument(
        "composition requires adjacent regions in one block and one A output");
  auto h = a.outputs[0];
  for (auto* user : h.getUsers())
    if (std::find(b.operations.begin(), b.operations.end(), user) ==
        b.operations.end())
      throw std::invalid_argument(
          "intermediate has a use outside B (fan-out/live-out)");
  PairInterface result;
  auto ai = values.interface(a), bi = values.interface(b);
  result.external.inputs = ai.inputs;
  result.external.outputs = bi.outputs;
  result.intermediate = ai.outputs[0];
  for (size_t i = 0; i < ai.inputs.size(); ++i) result.a_inputs.push_back(i);
  bool consumed = false;
  for (size_t i = 0; i < bi.inputs.size(); ++i) {
    auto port = bi.inputs[i];
    if (port.value == result.intermediate.value) {
      result.b_inputs.push_back(-1);
      result.intermediate_input = i;
      consumed = true;
      continue;
    }
    size_t index = 0;
    while (index < result.external.inputs.size() &&
           result.external.inputs[index].value != port.value)
      ++index;
    if (index == result.external.inputs.size())
      result.external.inputs.push_back(port);
    result.b_inputs.push_back(index);
  }
  if (!consumed) throw std::invalid_argument("B does not consume A output");
  return result;
}

}  // namespace joint_shard

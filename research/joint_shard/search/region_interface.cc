#include "research/joint_shard/search/region_interface.h"

#include <algorithm>
#include <stdexcept>

#include "mlir/Dialect/Func/IR/FuncOps.h"
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

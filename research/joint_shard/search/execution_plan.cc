#include "research/joint_shard/search/execution_plan.h"

#include <map>
#include <stdexcept>

#include "research/joint_shard/sharding/plan_materializer.h"

namespace joint_shard {
ExecutionPlanId ExecutionPlan::region(size_t index,
                                      const RegionSummary& summary,
                                      PlanId implementation) {
  const auto& selected = summary.plans.at(implementation);
  ExecutionPlanNode node;
  node.kind = PlanKind::Region;
  node.interface = summary.interface;
  node.boundary = selected.boundary;
  node.cost = selected.cost;
  node.region = index;
  node.implementation = implementation;
  nodes.push_back(std::move(node));
  return nodes.size() - 1;
}
ExecutionPlanId ExecutionPlan::adapter(const TensorPort& port,
                                       const ReshardPlan& selected) {
  if (!selected.feasible || !selected.cost.known() ||
      port.type != selected.type)
    throw std::invalid_argument("invalid executable adapter");
  if (selected.from == selected.to && selected.cost.total() != 0)
    throw std::invalid_argument("identity adapter must have zero cost");
  ExecutionPlanNode node;
  node.kind = PlanKind::Adapter;
  node.interface = {{port}, {port}};
  node.boundary = {{selected.from}, {selected.to}};
  node.cost = selected.cost;
  node.adapter = selected;
  nodes.push_back(std::move(node));
  return nodes.size() - 1;
}
ExecutionPlanId ExecutionPlan::composite(
    const RegionInterface& interface, const BoundaryState& boundary,
    std::vector<ExecutionPlanId> children) {
  ExecutionPlanNode node;
  node.interface = interface;
  node.boundary = boundary;
  node.children = std::move(children);
  for (auto child : node.children) node.cost += nodes.at(child).cost;
  nodes.push_back(std::move(node));
  return nodes.size() - 1;
}
ExecutionPlanId ExecutionPlan::resolved(size_t index,
                                        const RegionSummary& summary,
                                        const ResolvedRegionPlan& selected) {
  std::vector<ExecutionPlanId> children;
  for (size_t i = 0; i < selected.input_adapters.size(); ++i)
    if (selected.input_adapters[i].from != selected.input_adapters[i].to)
      children.push_back(
          adapter(summary.interface.inputs.at(i), selected.input_adapters[i]));
  children.push_back(region(index, summary, selected.implementation));
  for (size_t i = 0; i < selected.output_adapters.size(); ++i)
    if (selected.output_adapters[i].from != selected.output_adapters[i].to)
      children.push_back(adapter(summary.interface.outputs.at(i),
                                 selected.output_adapters[i]));
  return composite(summary.interface, selected.boundary, std::move(children));
}
namespace {
std::vector<mlir::Value> emit(PlanMaterializer& builder,
                              const ExecutionPlan& plan, ExecutionPlanId id,
                              llvm::ArrayRef<const RegionSummary*> regions,
                              mlir::ValueRange inputs) {
  const auto& node = plan.nodes.at(id);
  if (node.interface.inputs.size() != inputs.size())
    throw std::invalid_argument("execution plan input arity mismatch");
  if (node.kind == PlanKind::Region) {
    if (node.region >= regions.size() || !regions[node.region])
      throw std::invalid_argument(
          "execution plan region reference out of range");
    return builder.inlineArtifact(
        regions[node.region]->plans.at(node.implementation).lowered_mlir,
        inputs);
  }
  if (node.kind == PlanKind::Adapter) {
    if (node.adapter.from == node.adapter.to) return {inputs.front()};
    if (node.adapter.lowered_mlir.empty())
      throw std::invalid_argument("adapter lacks artifact");
    return builder.inlineArtifact(node.adapter.lowered_mlir, inputs);
  }
  // Each composite has its own bindings: adapting a region's argument cannot
  // change the external value supplied to another region using that argument.
  std::map<ValueId, mlir::Value> values;
  for (size_t i = 0; i < inputs.size(); ++i)
    values[node.interface.inputs[i].value] = inputs[i];
  for (auto child : node.children) {
    const auto& childNode = plan.nodes.at(child);
    std::vector<mlir::Value> arguments;
    for (auto port : childNode.interface.inputs)
      arguments.push_back(values.at(port.value));
    auto outputs = emit(builder, plan, child, regions, arguments);
    if (outputs.size() != childNode.interface.outputs.size())
      throw std::invalid_argument("execution plan output arity mismatch");
    for (size_t i = 0; i < outputs.size(); ++i)
      values[childNode.interface.outputs[i].value] = outputs[i];
  }
  std::vector<mlir::Value> outputs;
  for (auto port : node.interface.outputs)
    outputs.push_back(values.at(port.value));
  return outputs;
}
}  // namespace
std::string materializeExecutionPlan(
    const ExecutionPlan& plan, llvm::ArrayRef<const RegionSummary*> regions,
    const MeshContext& mesh, const CostModel& model) {
  const auto& root = plan.nodes.at(plan.root);
  std::vector<mlir::Type> inputs, outputs;
  for (auto port : root.interface.inputs) inputs.push_back(port.type);
  for (auto port : root.interface.outputs) outputs.push_back(port.type);
  PlanMaterializer builder(mesh, inputs, outputs, root.boundary);
  auto results = emit(builder, plan, plan.root, regions, builder.arguments());
  return builder.finish(results, root.cost, model);
}
std::string materializeExecutionPlan(const ExecutionPlan& plan,
                                     const std::vector<RegionSummary>& regions,
                                     const MeshContext& mesh,
                                     const CostModel& model) {
  std::vector<const RegionSummary*> references;
  for (const auto& region : regions) references.push_back(&region);
  return materializeExecutionPlan(plan, references, mesh, model);
}
}  // namespace joint_shard

#include "research/joint_shard/search/pair_composer.h"

#include <algorithm>
#include <stdexcept>

#include "research/joint_shard/search/execution_plan.h"

namespace joint_shard {

PairSummary composePair(const RegionSummary& a, const RegionSummary& b,
                        const PairInterface& interface,
                        const ReshardPlanner& oracle, size_t cap) {
  if (!cap) throw std::invalid_argument("pair evaluation cap must be positive");
  if (a.interface.outputs.size() != 1 ||
      interface.a_inputs.size() != a.interface.inputs.size() ||
      interface.b_inputs.size() != b.interface.inputs.size() ||
      interface.external.outputs.size() != b.interface.outputs.size() ||
      interface.intermediate_input >= b.interface.inputs.size() ||
      interface.b_inputs.at(interface.intermediate_input) != -1 ||
      a.interface.outputs[0].type != interface.intermediate.type ||
      b.interface.inputs[interface.intermediate_input].type !=
          interface.intermediate.type)
    throw std::invalid_argument("pair interface/type mismatch");
  if (a.interface.outputs[0] != interface.intermediate ||
      b.interface.inputs[interface.intermediate_input] !=
          interface.intermediate ||
      b.interface.outputs != interface.external.outputs)
    throw std::invalid_argument("pair SSA identity mismatch");
  std::vector<bool> bound(interface.external.inputs.size(), false);
  for (size_t i = 0; i < interface.a_inputs.size(); ++i) {
    auto binding = interface.a_inputs[i];
    if (binding >= bound.size() ||
        a.interface.inputs[i] != interface.external.inputs[binding])
      throw std::invalid_argument("invalid A external input binding");
    bound[binding] = true;
  }
  for (size_t i = 0; i < interface.b_inputs.size(); ++i) {
    auto binding = interface.b_inputs[i];
    if (binding == -1 && i == interface.intermediate_input) continue;
    if (binding < 0 || static_cast<size_t>(binding) >= bound.size() ||
        b.interface.inputs[i] != interface.external.inputs[binding])
      throw std::invalid_argument("invalid B external input binding");
    bound[binding] = true;
  }
  if (std::find(bound.begin(), bound.end(), false) != bound.end())
    throw std::invalid_argument("unbound pair external input");
  PairSummary result;
  result.a_region = a.id;
  result.b_region = b.id;
  result.interface = interface;
  std::vector<ResolvedRegionPlan> left, right;
  for (size_t i = 0; i < a.plans.size(); ++i)
    left.push_back(resolveRegionPlan(a, i, oracle));
  for (size_t i = 0; i < b.plans.size(); ++i)
    right.push_back(resolveRegionPlan(b, i, oracle));
  std::map<std::string, ComposedPlan> best;
  for (const auto& ap : left) {
    for (const auto& bp : right) {
      if (result.evaluations == cap) {
        result.truncated = true;
        break;
      }
      ++result.evaluations;
      BoundaryState boundary;
      boundary.inputs.resize(interface.external.inputs.size());
      boundary.outputs = bp.boundary.outputs;
      for (size_t i = 0; i < interface.a_inputs.size(); ++i)
        boundary.inputs.at(interface.a_inputs[i]) = ap.boundary.inputs.at(i);
      bool compatible = true;
      for (size_t i = 0; i < interface.b_inputs.size(); ++i) {
        auto binding = interface.b_inputs[i];
        if (binding < 0) continue;
        auto& external = boundary.inputs.at(binding);
        if (external.attr && external != bp.boundary.inputs.at(i)) {
          compatible = false;
          break;
        }
        external = bp.boundary.inputs.at(i);
      }
      if (!compatible) {
        ++result.shared_layout_rejections;
        continue;
      }
      auto adapter = oracle(ap.boundary.outputs[0],
                            bp.boundary.inputs[interface.intermediate_input],
                            interface.intermediate.type);
      Cost cost = ap.cost;
      cost += adapter.cost;
      cost += bp.cost;
      if (!cost.known() || !adapter.feasible) {
        ++result.unknown_cost_rejections;
        continue;
      }
      auto key = boundary.key();
      auto previous = best.find(key);
      if (previous == best.end() ||
          cost.total() < previous->second.cost.total() ||
          (cost.total() == previous->second.cost.total() &&
           std::make_pair(ap.requested, bp.requested) <
               std::make_pair(previous->second.a.requested,
                              previous->second.b.requested)))
        best.insert_or_assign(
            key,
            ComposedPlan{
                std::move(boundary), ap, bp, std::move(adapter), cost, {}});
    }
    if (result.truncated) break;
  }
  for (auto& [key, plan] : best) result.plans.push_back(std::move(plan));
  std::vector<BoundaryState> boundaries;
  std::vector<Cost> costs;
  for (const auto& plan : result.plans) {
    boundaries.push_back(plan.boundary);
    costs.push_back(plan.cost);
  }
  auto pruned = pruneBoundaryPlans(interface.external, boundaries, costs,
                                   [&](auto from, auto to, auto type) {
                                     auto adapter = oracle(from, to, type);
                                     return adapter.feasible ? adapter.cost
                                                             : Cost{0, 0, 1};
                                   });
  result.frontier = std::move(pruned.frontier);
  result.dominance = std::move(pruned.dominance);
  for (const auto& witness : result.dominance) {
    auto& plan = result.plans[witness.removed_id];
    const auto& core = result.plans[witness.replacement_id];
    plan.replacement = witness.replacement_id;
    plan.cost = core.cost;
    for (size_t i = 0; i < interface.external.inputs.size(); ++i) {
      auto adapter = oracle(plan.boundary.inputs[i], core.boundary.inputs[i],
                            interface.external.inputs[i].type);
      plan.cost += adapter.cost;
      plan.input_adapters.push_back(std::move(adapter));
    }
    for (size_t i = 0; i < interface.external.outputs.size(); ++i) {
      auto adapter = oracle(core.boundary.outputs[i], plan.boundary.outputs[i],
                            interface.external.outputs[i].type);
      plan.cost += adapter.cost;
      plan.output_adapters.push_back(std::move(adapter));
    }
    plan.a = {};
    plan.b = {};
    plan.intermediate = {};
  }
  return result;
}
std::string materializePair(const RegionSummary& a, const RegionSummary& b,
                            const PairSummary& pair, PlanId requested,
                            const MeshContext& mesh, const CostModel& model) {
  const auto& plan = pair.plans.at(requested);
  const auto& core = plan.replacement ? pair.plans.at(*plan.replacement) : plan;
  ExecutionPlan execution;
  std::vector<ExecutionPlanId> children;
  for (size_t i = 0; i < plan.input_adapters.size(); ++i)
    if (plan.input_adapters[i].from != plan.input_adapters[i].to)
      children.push_back(execution.adapter(pair.interface.external.inputs[i],
                                           plan.input_adapters[i]));
  std::vector<ExecutionPlanId> local{execution.resolved(0, a, core.a)};
  if (core.intermediate.from != core.intermediate.to)
    local.push_back(
        execution.adapter(pair.interface.intermediate, core.intermediate));
  local.push_back(execution.resolved(1, b, core.b));
  children.push_back(execution.composite(pair.interface.external, core.boundary,
                                         std::move(local)));
  for (size_t i = 0; i < plan.output_adapters.size(); ++i)
    if (plan.output_adapters[i].from != plan.output_adapters[i].to)
      children.push_back(execution.adapter(pair.interface.external.outputs[i],
                                           plan.output_adapters[i]));
  execution.root = execution.composite(pair.interface.external, plan.boundary,
                                       std::move(children));
  const auto& cost = execution.nodes.at(execution.root).cost;
  if (std::abs(cost.compute - plan.cost.compute) > 1e-9 ||
      std::abs(cost.communication - plan.cost.communication) > 1e-9)
    throw std::runtime_error(
        "selected pair cost disagrees with execution plan");
  const RegionSummary* regions[] = {&a, &b};
  return materializeExecutionPlan(execution, llvm::ArrayRef(regions), mesh,
                                  model);
}

}  // namespace joint_shard

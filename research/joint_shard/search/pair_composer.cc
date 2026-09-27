#include "research/joint_shard/search/pair_composer.h"

#include <algorithm>
#include <stdexcept>

#include "research/joint_shard/sharding/plan_materializer.h"

namespace joint_shard {

ResolvedRegionPlan resolveRegionPlan(const RegionSummary& region,
                                     PlanId requested, bool use_frontier,
                                     const ReshardPlanner& oracle) {
  const auto& exact = region.plans.at(requested);
  ResolvedRegionPlan result;
  result.requested = requested;
  result.implementation = requested;
  result.boundary = exact.boundary;
  if (use_frontier) {
    for (const auto& witness : region.dominance)
      if (witness.removed_id == requested) {
        result.implementation = witness.replacement_id;
        break;
      }
    if (std::find(region.frontier.begin(), region.frontier.end(),
                  result.implementation) == region.frontier.end())
      throw std::invalid_argument(
          "dominance witness does not resolve directly to frontier");
  }
  const auto& core = region.plans.at(result.implementation);
  result.candidate_id = core.candidate_id;
  result.core_cost = core.cost;
  for (size_t i = 0; i < exact.boundary.inputs.size(); ++i) {
    auto adapter = oracle(exact.boundary.inputs[i], core.boundary.inputs[i],
                          region.interface.inputs.at(i).type);
    result.adapters_cost += adapter.cost;
    if (!adapter.feasible) ++result.adapters_cost.unknown;
    result.input_adapters.push_back(std::move(adapter));
  }
  for (size_t i = 0; i < exact.boundary.outputs.size(); ++i) {
    auto adapter = oracle(core.boundary.outputs[i], exact.boundary.outputs[i],
                          region.interface.outputs.at(i).type);
    result.adapters_cost += adapter.cost;
    if (!adapter.feasible) ++result.adapters_cost.unknown;
    result.output_adapters.push_back(std::move(adapter));
  }
  result.cost = result.core_cost;
  result.cost += result.adapters_cost;
  if (use_frontier && exact.cost.known() && result.cost.known() &&
      result.cost.total() > exact.cost.total() + 1e-9)
    throw std::invalid_argument(
        "resolved plan exceeds exact cost (inconsistent oracle)");
  return result;
}
PairSummary composePair(const RegionSummary& a, const RegionSummary& b,
                        const PairInterface& interface,
                        const ReshardPlanner& oracle, size_t cap,
                        bool use_frontier) {
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
  result.frontier_resolved = use_frontier;
  std::vector<ResolvedRegionPlan> left, right;
  for (size_t i = 0; i < a.plans.size(); ++i)
    left.push_back(resolveRegionPlan(a, i, use_frontier, oracle));
  for (size_t i = 0; i < b.plans.size(); ++i)
    right.push_back(resolveRegionPlan(b, i, use_frontier, oracle));
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
  return result;
}
namespace {
mlir::Value adapt(PlanMaterializer& builder, const ReshardPlan& adapter,
                  mlir::Value input) {
  if (!adapter.feasible)
    throw std::invalid_argument("cannot materialize infeasible adapter");
  if (adapter.from == adapter.to) return input;
  if (adapter.lowered_mlir.empty())
    throw std::invalid_argument("nonidentity adapter lacks artifact");
  return builder.inlineArtifact(adapter.lowered_mlir, {input}).at(0);
}
std::vector<mlir::Value> inlineRegion(PlanMaterializer& builder,
                                      const RegionSummary& summary,
                                      const ResolvedRegionPlan& plan,
                                      std::vector<mlir::Value> inputs) {
  for (size_t i = 0; i < inputs.size(); ++i)
    inputs[i] = adapt(builder, plan.input_adapters.at(i), inputs[i]);
  auto outputs = builder.inlineArtifact(
      summary.plans.at(plan.implementation).lowered_mlir, inputs);
  for (size_t i = 0; i < outputs.size(); ++i)
    outputs[i] = adapt(builder, plan.output_adapters.at(i), outputs[i]);
  return outputs;
}
}  // namespace
std::string materializePair(const RegionSummary& a, const RegionSummary& b,
                            const PairInterface& interface,
                            const ComposedPlan& plan, const MeshContext& mesh,
                            const CostModel& model) {
  std::vector<mlir::Type> inputs, outputs;
  for (auto port : interface.external.inputs) inputs.push_back(port.type);
  for (auto port : interface.external.outputs) outputs.push_back(port.type);
  PlanMaterializer builder(mesh, inputs, outputs, plan.boundary);
  std::vector<mlir::Value> ai, bi;
  for (auto i : interface.a_inputs) ai.push_back(builder.arguments()[i]);
  auto av = inlineRegion(builder, a, plan.a, std::move(ai));
  auto h = adapt(builder, plan.intermediate, av.at(0));
  for (auto i : interface.b_inputs)
    bi.push_back(i < 0 ? h : builder.arguments()[i]);
  auto bv = inlineRegion(builder, b, plan.b, std::move(bi));
  return builder.finish(bv, plan.cost, model);
}

}  // namespace joint_shard

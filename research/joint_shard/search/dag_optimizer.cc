#include "research/joint_shard/search/dag_optimizer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <stdexcept>

namespace joint_shard {
namespace {
struct Prefix {
  Cost cost;
  std::vector<TensorSharding> live;
  ResolvedRegionPlan selected;
  std::vector<ReshardPlan> incoming;
  size_t predecessor = 0;
  std::vector<PlanId> path;
};
bool better(const Prefix& a, const Prefix& b) {
  return a.cost.total() < b.cost.total() ||
         (a.cost.total() == b.cost.total() && a.path < b.path);
}
using Layouts = std::map<ValueId, TensorSharding>;
Layouts externalLayouts(const DagInterface& interface,
                        const BoundaryState& contract) {
  Layouts result;
  for (size_t i = 0; i < contract.inputs.size(); ++i)
    result.emplace(interface.external.inputs[i].value, contract.inputs[i]);
  return result;
}
Layouts returnLayouts(const DagInterface& interface,
                      const BoundaryState& contract) {
  Layouts result;
  for (size_t i = 0; i < contract.outputs.size(); ++i) {
    auto [it, inserted] = result.emplace(interface.external.outputs[i].value,
                                         contract.outputs[i]);
    if (!inserted && it->second != contract.outputs[i])
      throw std::invalid_argument(
          "DAG repeated return value has inconsistent contracts");
  }
  return result;
}
}  // namespace
DagResult optimizeDag(const std::vector<RegionSummary>& regions,
                      const DagInterface& interface,
                      const BoundaryState& contract,
                      const ReshardPlanner& oracle,
                      const DagSearchOptions& options) {
  if (!options.max_live_values || !options.max_states ||
      !options.max_transitions || regions.empty() ||
      regions.size() != interface.regions.size() ||
      contract.inputs.size() != interface.external.inputs.size() ||
      contract.outputs.size() != interface.external.outputs.size())
    throw std::invalid_argument(
        "invalid DAG interface, contract or search caps");
  if (buildLiveCuts(interface.external, interface.regions) !=
      interface.live_after)
    throw std::invalid_argument("DAG live cuts do not match dependencies");
  DagResult result;
  result.mode = options.mode;
  result.contract = contract;
  for (size_t i = 0; i < regions.size(); ++i) {
    if (regions[i].interface.inputs != interface.regions[i].inputs ||
        regions[i].interface.outputs != interface.regions[i].outputs)
      throw std::invalid_argument("DAG summary interface mismatch");
    result.peak_live_values =
        std::max(result.peak_live_values, interface.live_after[i].size());
  }
  if (result.peak_live_values > options.max_live_values)
    throw std::invalid_argument(
        "DAG peak live values " + std::to_string(result.peak_live_values) +
        " exceed max-live-values=" + std::to_string(options.max_live_values));
  const auto external = externalLayouts(interface, contract);
  const auto returns = returnLayouts(interface, contract);
  for (const auto& [value, layout] : returns)
    if (auto found = external.find(value);
        found != external.end() && found->second != layout)
      throw std::invalid_argument(
          "DAG returned function argument has incompatible layouts");
  const auto start = std::chrono::steady_clock::now();
  auto elapsed = [&]() {
    return std::chrono::duration<double, std::micro>(
               std::chrono::steady_clock::now() - start)
        .count();
  };
  std::vector<std::vector<Prefix>> layers;
  const std::vector<Prefix> initial(1);
  for (size_t i = 0; i < regions.size(); ++i) {
    const auto& region = regions[i];
    std::vector<ResolvedRegionPlan> choices;
    for (const auto& plan : region.plans) {
      if (plan.id >= region.plans.size() || &region.plans[plan.id] != &plan)
        throw std::invalid_argument(
            "DAG plans must be finalized before search");
      if (plan.boundary.inputs.size() != region.interface.inputs.size() ||
          plan.boundary.outputs.size() != region.interface.outputs.size())
        throw std::invalid_argument("DAG plan boundary arity mismatch");
      bool compatible = true;
      for (size_t j = 0; j < region.interface.inputs.size(); ++j)
        if (auto found = external.find(region.interface.inputs[j].value);
            found != external.end() && plan.boundary.inputs[j] != found->second)
          compatible = false;
      for (size_t j = 0; j < region.interface.outputs.size(); ++j)
        if (auto found = returns.find(region.interface.outputs[j].value);
            found != returns.end() && plan.boundary.outputs[j] != found->second)
          compatible = false;
      if (!compatible) {
        ++result.contract_rejections;
        continue;
      }
      auto selected = resolveRegionPlan(
          region, plan.id, options.mode == DagSearchMode::Resolved, oracle);
      if (!selected.cost.known()) {
        ++result.unknown_cost_rejections;
        continue;
      }
      choices.push_back(std::move(selected));
    }
    const auto& previousLayer = i ? layers.back() : initial;
    std::map<std::string, Prefix> best;
    size_t attempted = 0;
    bool transitionCap = false;
    for (size_t p = 0; p < previousLayer.size(); ++p) {
      const auto& prefix = previousLayer[p];
      Layouts available = external;
      if (i)
        for (size_t j = 0; j < prefix.live.size(); ++j)
          available.emplace(interface.live_after[i - 1][j].value,
                            prefix.live[j]);
      for (const auto& choice : choices) {
        if (attempted == options.max_transitions) {
          transitionCap = true;
          break;
        }
        ++attempted;
        ++result.transitions;
        Prefix next;
        next.cost = prefix.cost;
        next.selected = choice;
        next.predecessor = p;
        next.path = prefix.path;
        bool feasible = true;
        for (size_t j = 0; j < region.interface.inputs.size(); ++j) {
          auto port = region.interface.inputs[j];
          auto from = available.at(port.value), to = choice.boundary.inputs[j];
          auto adapter = oracle(from, to, port.type);
          if (!adapter.feasible || !adapter.cost.known()) {
            feasible = false;
            break;
          }
          if (adapter.from != from || adapter.to != to ||
              adapter.type != port.type ||
              (from == to && adapter.cost.total() != 0))
            throw std::invalid_argument("inconsistent DAG adapter oracle");
          next.cost += adapter.cost;
          next.incoming.push_back(std::move(adapter));
        }
        if (!feasible) {
          ++result.unknown_cost_rejections;
          continue;
        }
        next.cost += choice.cost;
        next.path.push_back(choice.requested);
        // Only produced outputs change the persistent live-value environment.
        // Consumer adaptations are temporary copies, never persistent updates.
        auto after = available;
        for (size_t j = 0; j < region.interface.outputs.size(); ++j)
          after.emplace(region.interface.outputs[j].value,
                        choice.boundary.outputs[j]);
        for (auto port : interface.live_after[i])
          next.live.push_back(after.at(port.value));
        auto key = options.mode == DagSearchMode::Greedy
                       ? std::string{}
                       : BoundaryState{{}, next.live}.key();
        auto found = best.find(key);
        if (found == best.end() || better(next, found->second))
          best.insert_or_assign(key, std::move(next));
      }
      if (transitionCap) break;
    }
    result.truncated |= transitionCap;
    result.transitions_truncated_per_layer.push_back(transitionCap);
    if (best.empty()) {
      result.failure =
          "no feasible DAG prefix at region " + std::to_string(region.id);
      result.search_us = elapsed();
      return result;
    }
    std::vector<Prefix> retained;
    for (auto& [key, prefix] : best) retained.push_back(std::move(prefix));
    size_t discarded = 0;
    if (retained.size() > options.max_states) {
      std::sort(retained.begin(), retained.end(), better);
      discarded = retained.size() - options.max_states;
      retained.resize(options.max_states);
      result.truncated = true;
    }
    result.states_discarded += discarded;
    result.discarded_per_layer.push_back(discarded);
    result.states_retained += retained.size();
    result.states_per_layer.push_back(retained.size());
    layers.push_back(std::move(retained));
  }
  auto winner =
      std::min_element(layers.back().begin(), layers.back().end(), better);
  result.cost = winner->cost;
  size_t index = winner - layers.back().begin();
  for (size_t i = regions.size(); i-- > 0;) {
    const auto& prefix = layers[i][index];
    result.steps.push_back({i, prefix.selected, prefix.incoming, prefix.live});
    index = prefix.predecessor;
  }
  std::reverse(result.steps.begin(), result.steps.end());
  for (size_t i = 0; i < result.steps.size(); ++i) {
    const auto& step = result.steps[i];
    std::vector<ExecutionPlanId> local;
    BoundaryState consumerBoundary;
    for (size_t j = 0; j < step.incoming.size(); ++j) {
      consumerBoundary.inputs.push_back(step.incoming[j].from);
      if (step.incoming[j].from != step.incoming[j].to)
        local.push_back(result.execution.adapter(regions[i].interface.inputs[j],
                                                 step.incoming[j]));
    }
    local.push_back(result.execution.resolved(i, regions[i], step.selected));
    consumerBoundary.outputs = step.selected.boundary.outputs;
    // Adapter nodes rewrite their ValueId only within this consumer composite.
    auto consumer = result.execution.composite(
        regions[i].interface, consumerBoundary, std::move(local));
    std::vector<ExecutionPlanId> children;
    if (i) children.push_back(result.execution.root);
    children.push_back(consumer);
    result.execution.root = result.execution.composite(
        {interface.external.inputs, interface.live_after[i]},
        {contract.inputs, step.live_layouts}, std::move(children));
  }
  result.execution.root = result.execution.composite(
      interface.external, contract, {result.execution.root});
  const auto& actual = result.execution.nodes[result.execution.root].cost;
  auto close = [](double a, double b) {
    return std::abs(a - b) <= 1e-9 * std::max({1.0, std::abs(a), std::abs(b)});
  };
  if (!close(actual.compute, result.cost.compute) ||
      !close(actual.communication, result.cost.communication))
    throw std::runtime_error(
        "DAG reconstruction cost differs from selected prefix");
  result.feasible = true;
  result.search_us = elapsed();
  return result;
}
}  // namespace joint_shard

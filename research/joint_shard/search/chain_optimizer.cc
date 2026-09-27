#include "research/joint_shard/search/chain_optimizer.h"

#include <algorithm>
#include <chrono>
#include <map>
#include <stdexcept>

namespace joint_shard {
namespace {
struct Prefix {
  Cost cost;
  ResolvedRegionPlan selected;
  std::optional<ReshardPlan> incoming;
  size_t predecessor = 0;
  std::vector<PlanId> path;
};
bool better(const Prefix& a, const Prefix& b) {
  return a.cost.total() < b.cost.total() ||
         (a.cost.total() == b.cost.total() && a.path < b.path);
}
std::string outgoingKey(const BoundaryState& boundary) {
  return BoundaryState{{}, boundary.outputs}.key();
}
}  // namespace
ChainResult optimizeChain(const std::vector<RegionSummary>& regions,
                          const ChainInterface& interface,
                          const BoundaryState& contract,
                          const ReshardPlanner& oracle,
                          const ChainSearchOptions& options) {
  if (!options.max_transitions || regions.empty() ||
      regions.size() != interface.regions.size() ||
      regions.size() != interface.inputs.size() ||
      regions.size() != interface.intermediate_inputs.size() ||
      interface.external.inputs.size() != contract.inputs.size() ||
      interface.external.outputs.size() != contract.outputs.size())
    throw std::invalid_argument(
        "invalid chain interface, contract or search cap");
  for (size_t i = 0; i < regions.size(); ++i) {
    if (regions[i].interface.inputs != interface.regions[i].inputs ||
        regions[i].interface.outputs != interface.regions[i].outputs ||
        interface.inputs[i].size() != regions[i].interface.inputs.size() ||
        (i + 1 < regions.size() && regions[i].interface.outputs.size() != 1))
      throw std::invalid_argument("chain summary interface mismatch");
    size_t intermediateCount = 0;
    for (size_t j = 0; j < interface.inputs[i].size(); ++j) {
      auto binding = interface.inputs[i][j];
      if (binding == -1) {
        if (!i || j != interface.intermediate_inputs[i] ||
            regions[i].interface.inputs[j] !=
                regions[i - 1].interface.outputs.at(0))
          throw std::invalid_argument("invalid chain intermediate binding");
        ++intermediateCount;
      } else if (binding < 0 ||
                 static_cast<size_t>(binding) >=
                     interface.external.inputs.size() ||
                 regions[i].interface.inputs[j] !=
                     interface.external.inputs[binding])
        throw std::invalid_argument("invalid chain external binding");
    }
    if (intermediateCount != (i ? 1 : 0))
      throw std::invalid_argument("chain must have one intermediate per cut");
  }
  ChainResult result;
  result.mode = options.mode;
  result.contract = contract;
  const auto start = std::chrono::steady_clock::now();
  auto elapsed = [&]() {
    return std::chrono::duration<double, std::micro>(
               std::chrono::steady_clock::now() - start)
        .count();
  };
  std::vector<std::vector<Prefix>> layers;
  for (size_t i = 0; i < regions.size(); ++i) {
    std::vector<ResolvedRegionPlan> choices;
    for (const auto& plan : regions[i].plans) {
      if (plan.id >= regions[i].plans.size() ||
          &regions[i].plans[plan.id] != &plan)
        throw std::invalid_argument(
            "chain plans must be finalized before search");
      if (plan.boundary.inputs.size() != regions[i].interface.inputs.size() ||
          plan.boundary.outputs.size() != regions[i].interface.outputs.size())
        throw std::invalid_argument("chain plan boundary arity mismatch");
      bool compatible = true;
      for (size_t j = 0; j < interface.inputs[i].size(); ++j) {
        auto binding = interface.inputs[i][j];
        if (binding >= 0 && plan.boundary.inputs[j] != contract.inputs[binding])
          compatible = false;
      }
      if (i + 1 == regions.size()) {
        for (size_t j = 0; j < interface.external.outputs.size(); ++j) {
          auto found = std::find(regions[i].interface.outputs.begin(),
                                 regions[i].interface.outputs.end(),
                                 interface.external.outputs[j]);
          if (found == regions[i].interface.outputs.end())
            throw std::invalid_argument("invalid chain return binding");
          if (plan.boundary
                  .outputs[found - regions[i].interface.outputs.begin()] !=
              contract.outputs[j])
            compatible = false;
        }
      }
      if (!compatible) {
        ++result.contract_rejections;
        continue;
      }
      auto resolved =
          resolveRegionPlan(regions[i], plan.id,
                            options.mode == ChainSearchMode::Resolved, oracle);
      if (!resolved.cost.known()) {
        ++result.unknown_cost_rejections;
        continue;
      }
      choices.push_back(std::move(resolved));
    }
    std::map<std::string, Prefix> best;
    size_t attempted = 0;
    size_t count = i ? layers.back().size() : 1;
    for (size_t previous = 0; previous < count; ++previous) {
      for (const auto& choice : choices) {
        if (attempted == options.max_transitions) {
          result.truncated = true;
          break;
        }
        ++attempted;
        ++result.transitions;
        Prefix next;
        next.selected = choice;
        next.predecessor = previous;
        if (i) {
          const auto& prefix = layers.back()[previous];
          next.cost += prefix.cost;
          next.path = prefix.path;
          auto adapter =
              oracle(prefix.selected.boundary.outputs.at(0),
                     choice.boundary.inputs[interface.intermediate_inputs[i]],
                     regions[i - 1].interface.outputs[0].type);
          if (!adapter.feasible || !adapter.cost.known()) {
            ++result.unknown_cost_rejections;
            continue;
          }
          next.cost += adapter.cost;
          next.incoming = std::move(adapter);
        }
        next.cost += choice.cost;
        next.path.push_back(choice.requested);
        auto key = options.mode == ChainSearchMode::Greedy
                       ? std::string{}
                       : outgoingKey(choice.boundary);
        auto found = best.find(key);
        if (found == best.end() || better(next, found->second))
          best.insert_or_assign(key, std::move(next));
      }
      if (attempted == options.max_transitions && previous + 1 < count) {
        result.truncated = true;
        break;
      }
    }
    if (best.empty()) {
      result.failure =
          "no feasible prefix at region " + std::to_string(regions[i].id);
      result.search_us = elapsed();
      return result;
    }
    std::vector<Prefix> retained;
    for (auto& [key, prefix] : best) retained.push_back(std::move(prefix));
    result.states_retained += retained.size();
    result.states_per_layer.push_back(retained.size());
    layers.push_back(std::move(retained));
  }
  auto winner =
      std::min_element(layers.back().begin(), layers.back().end(), better);
  size_t index = winner - layers.back().begin();
  result.cost = winner->cost;
  for (size_t i = regions.size(); i-- > 0;) {
    const auto& prefix = layers[i][index];
    result.steps.push_back({i, prefix.selected, prefix.incoming});
    index = prefix.predecessor;
  }
  std::reverse(result.steps.begin(), result.steps.end());
  // Reconstruct just the winner. Prefix composites can themselves be children.
  for (size_t i = 0; i < result.steps.size(); ++i) {
    const auto& step = result.steps[i];
    std::vector<ExecutionPlanId> children;
    if (i) {
      children.push_back(result.execution.root);
      if (step.incoming->from != step.incoming->to)
        children.push_back(result.execution.adapter(
            regions[i - 1].interface.outputs[0], *step.incoming));
    }
    children.push_back(result.execution.resolved(i, regions[i], step.selected));
    RegionInterface prefixInterface{interface.external.inputs,
                                    regions[i].interface.outputs};
    BoundaryState prefixBoundary{contract.inputs,
                                 step.selected.boundary.outputs};
    result.execution.root = result.execution.composite(
        prefixInterface, prefixBoundary, std::move(children));
  }
  result.execution.root = result.execution.composite(
      interface.external, contract, {result.execution.root});
  const auto& executable = result.execution.nodes[result.execution.root].cost;
  auto close = [](double a, double b) {
    return std::abs(a - b) <= 1e-9 * std::max({1.0, std::abs(a), std::abs(b)});
  };
  if (!close(executable.compute, result.cost.compute) ||
      !close(executable.communication, result.cost.communication))
    throw std::runtime_error(
        "chain reconstruction cost disagrees with selected prefix");
  result.feasible = true;
  result.search_us = elapsed();
  return result;
}
}  // namespace joint_shard

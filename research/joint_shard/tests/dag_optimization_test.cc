#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <random>

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "gtest/gtest.h"
#include "research/joint_shard/reporting/artifact_writer.h"
#include "research/joint_shard/reporting/reports.h"
#include "research/joint_shard/search/region_optimizer.h"
#include "research/joint_shard/sharding/region_evaluator.h"
#include "research/joint_shard/tests/support/fixture_interpreter.h"
#include "research/joint_shard/tests/support/mlir_test.h"
#include "shardy/dialect/sdy/ir/utils.h"

namespace joint_shard {
namespace {
class DagOptimizationTest : public test::MlirTest {
 protected:
  mlir::OwningOpRef<mlir::ModuleOp> fixture(const std::string& name) {
    auto module = mlir::parseSourceFile<mlir::ModuleOp>(
        "research/joint_shard/testdata/" + name + ".mlir", &context);
    if (!module) throw std::runtime_error("missing DAG fixture");
    return module;
  }
  DagInterface describe(mlir::ModuleOp module) {
    auto main = module.lookupSymbol<mlir::func::FuncOp>("main");
    return buildDagInterface(main, Regionizer().split(main),
                             ValueIndex(module));
  }
  std::vector<ValueId> ids(const std::vector<TensorPort>& cut) {
    std::vector<ValueId> result;
    for (auto port : cut) result.push_back(port.value);
    return result;
  }
  // Independent complete Cartesian enumeration. Layouts are recorded once at
  // production; consumer adapters never update the producer's layout.
  double exhaustive(const std::vector<RegionSummary>& regions,
                    const DagInterface& interface,
                    const BoundaryState& contract,
                    const ReshardPlanner& oracle) {
    double best = std::numeric_limits<double>::infinity();
    std::vector<const RegionPlan*> selected;
    std::function<void(size_t)> visit = [&](size_t index) {
      if (index < regions.size()) {
        for (const auto& plan : regions[index].plans) {
          selected.push_back(&plan);
          visit(index + 1);
          selected.pop_back();
        }
        return;
      }
      Cost cost;
      std::map<ValueId, TensorSharding> produced;
      for (size_t j = 0; j < interface.external.inputs.size(); ++j)
        produced.emplace(interface.external.inputs[j].value,
                         contract.inputs[j]);
      for (size_t i = 0; i < regions.size(); ++i) {
        const auto& plan = *selected[i];
        for (size_t j = 0; j < regions[i].interface.inputs.size(); ++j) {
          auto port = regions[i].interface.inputs[j];
          auto argument = std::find(interface.external.inputs.begin(),
                                    interface.external.inputs.end(), port);
          if (argument != interface.external.inputs.end() &&
              plan.boundary.inputs[j] !=
                  contract.inputs[argument - interface.external.inputs.begin()])
            return;
          auto adapter = oracle(produced.at(port.value),
                                plan.boundary.inputs[j], port.type);
          if (!adapter.feasible) return;
          cost += adapter.cost;
        }
        cost += plan.cost;
        for (size_t j = 0; j < regions[i].interface.outputs.size(); ++j)
          produced.emplace(regions[i].interface.outputs[j].value,
                           plan.boundary.outputs[j]);
      }
      for (size_t j = 0; j < interface.external.outputs.size(); ++j)
        if (produced.at(interface.external.outputs[j].value) !=
            contract.outputs[j])
          return;
      if (cost.known()) best = std::min(best, cost.total());
    };
    visit(0);
    return best;
  }
  void compare(const std::vector<test::DenseTensor>& expected,
               const std::vector<test::DenseTensor>& actual) {
    ASSERT_EQ(expected.size(), actual.size());
    constexpr double atol = 2e-5, rtol = 2e-4;
    for (size_t i = 0; i < expected.size(); ++i) {
      ASSERT_EQ(expected[i].size(), actual[i].size());
      for (size_t j = 0; j < expected[i].size(); ++j) {
        ASSERT_TRUE(std::isfinite(actual[i][j]));
        EXPECT_NEAR(actual[i][j], expected[i][j],
                    atol + rtol * std::abs(expected[i][j]))
            << "result=" << i << " element=" << j;
      }
    }
  }
  std::vector<test::DenseTensor> inputs(mlir::ModuleOp module,
                                        std::mt19937& random,
                                        bool cancellation = false) {
    std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
    std::vector<test::DenseTensor> result;
    auto main = module.lookupSymbol<mlir::func::FuncOp>("main");
    for (auto argument : main.getArguments()) {
      auto type = llvm::cast<mlir::RankedTensorType>(argument.getType());
      test::DenseTensor tensor(type.getNumElements());
      for (size_t i = 0; i < tensor.size(); ++i)
        tensor[i] = type.getRank() == 0 ? 0.3f
                    : cancellation      ? (i % 2 ? -0.5f : 0.5f)
                                        : distribution(random);
      result.push_back(std::move(tensor));
    }
    return result;
  }
  void verifyCostAndOccurrences(const DagResult& result,
                                const OptimizationReport& report,
                                const CostModel& model) {
    auto emitted =
        parse(result.lowered_mlir.empty()
                  ? materializeExecutionPlan(result.execution, report.regions,
                                             report.mesh, model)
                  : result.lowered_mlir);
    auto cost = model.estimate(*emitted, report.mesh.mesh);
    EXPECT_NEAR(cost.compute, result.cost.compute, 1e-9);
    EXPECT_NEAR(cost.communication, result.cost.communication, 1e-9);
    size_t expectedOperations = 0, regionLeaves = 0;
    for (const auto& node : result.execution.nodes) {
      std::string text;
      if (node.kind == PlanKind::Region) {
        ++regionLeaves;
        text =
            report.regions[node.region].plans[node.implementation].lowered_mlir;
      } else if (node.kind == PlanKind::Adapter)
        text = node.adapter.lowered_mlir;
      if (text.empty()) continue;
      auto leaf = parse(text);
      expectedOperations += leaf->lookupSymbol<mlir::func::FuncOp>("main")
                                .getBody()
                                .front()
                                .getOperations()
                                .size() -
                            1;
    }
    EXPECT_EQ(regionLeaves, report.regions.size());
    auto main = emitted->lookupSymbol<mlir::func::FuncOp>("main");
    EXPECT_EQ(main.getBody().front().getOperations().size() - 1,
              expectedOperations);
  }
};

TEST_F(DagOptimizationTest,
       LiveCutsTrackResidualsFanoutAndIndependentBranches) {
  auto residual = fixture("residual_block");
  auto graph = describe(*residual);
  ASSERT_EQ(graph.live_after.size(), 3);
  EXPECT_EQ(ids(graph.live_after[0]), (std::vector<ValueId>{6}));
  EXPECT_EQ(ids(graph.live_after[1]), (std::vector<ValueId>{6, 7}));
  EXPECT_EQ(ids(graph.live_after[2]), (std::vector<ValueId>{9}));
  EXPECT_EQ(graph.values.at(6).producer, 0);
  EXPECT_EQ(graph.values.at(6).consumers, (std::vector<size_t>{1, 2}));
  auto fanout = fixture("fanout_join");
  graph = describe(*fanout);
  EXPECT_EQ(ids(graph.live_after[0]), (std::vector<ValueId>{2}));
  EXPECT_EQ(ids(graph.live_after[1]), (std::vector<ValueId>{2, 3}));
  EXPECT_EQ(ids(graph.live_after[2]), (std::vector<ValueId>{3, 4}));
  EXPECT_EQ(graph.values.at(2).consumers, (std::vector<size_t>{1, 2}));
  auto independent = fixture("independent_branches");
  graph = describe(*independent);
  EXPECT_EQ(ids(graph.live_after[1]), (std::vector<ValueId>{2, 3}));
  EXPECT_EQ(graph.regions[1].inputs,
            (std::vector<TensorPort>{graph.external.inputs[1]}));
  auto multiple = fixture("dag_multi_output");
  graph = describe(*multiple);
  EXPECT_EQ(graph.regions[0].outputs.size(), 2);
  EXPECT_EQ(ids(graph.live_after[2]), (std::vector<ValueId>{3, 5}));
  EXPECT_EQ(graph.values.at(3).consumers, (std::vector<size_t>{2, 3}));
  EXPECT_EQ(ids(graph.external.outputs), (std::vector<ValueId>{5, 3, 5, 0}));
}

TEST_F(DagOptimizationTest, RealGraphsMatchExhaustiveAndPreserveSource) {
  for (const auto& name : {"residual_block", "fanout_join",
                           "independent_branches", "dag_multi_output"}) {
    SCOPED_TRACE(name);
    auto module = fixture(name);
    auto before = print(*module);
    RegionOptimizerOptions options;
    options.optimize_dag = true;
    auto report = summarizeRegions(*module, options);
    ASSERT_TRUE(report.dag);
    const auto& dag = *report.dag;
    EXPECT_EQ(print(*module), before);
    EXPECT_FALSE(dag.boundary_truncated);
    EXPECT_FALSE(dag.exact.truncated);
    ReshardCostOracle oracle(report.mesh);
    ReshardPlanner planner = [&](auto from, auto to, auto type) {
      return oracle.plan(from, to, type);
    };
    EXPECT_NEAR(
        dag.exact.cost.total(),
        exhaustive(report.regions, dag.interface, dag.contract, planner), 1e-9);
    EXPECT_LE(dag.resolved.cost.total(), dag.exact.cost.total() + 1e-9);
    EXPECT_LE(dag.exact.cost.total(), dag.original.cost.total() + 1e-9);
    verifyCostAndOccurrences(dag.exact, report, CostModel{});
    verifyCostAndOccurrences(dag.resolved, report, CostModel{});
    auto original = parse(materializeExecutionPlan(dag.original.execution,
                                                   report.original_regions,
                                                   report.mesh, CostModel{}));
    EXPECT_NEAR(CostModel().estimate(*original, report.mesh.mesh).total(),
                dag.original.cost.total(), 1e-9);
    EXPECT_NE(formatReport(dag).find("Live after region"), std::string::npos);
    EXPECT_NE(formatComparison(dag).find("Peak live values: 2"),
              std::string::npos);
  }
}

TEST_F(DagOptimizationTest, DAGAndChainAgreeForThreeToFiveRegions) {
  for (size_t n : {3, 4, 5}) {
    auto module = fixture("chain_" + std::to_string(n));
    RegionOptimizerOptions options;
    options.optimize_chain = true;
    auto chain = summarizeRegions(*module, options);
    options.optimize_chain = false;
    options.optimize_dag = true;
    auto dag = summarizeRegions(*module, options);
    EXPECT_NEAR(dag.dag->exact.cost.total(), chain.chain->exact.cost.total(),
                1e-9);
    EXPECT_NEAR(dag.dag->resolved.cost.total(),
                chain.chain->resolved.cost.total(), 1e-9);
    EXPECT_NEAR(dag.dag->original.cost.total(),
                chain.chain->original.cost.total(), 1e-9);
    EXPECT_EQ(dag.dag->exact.states_per_layer,
              chain.chain->exact.states_per_layer);
  }
}

TEST_F(DagOptimizationTest, SyntheticResidualBeatsGreedyAndBoundsAreExplicit) {
  auto module = fixture("residual_block");
  auto mesh = selectMesh(*module);
  auto type =
      mlir::RankedTensorType::get({8, 4}, mlir::Float32Type::get(&context));
  auto layouts = tensorLayoutChoices(type, mesh, {}, {});
  auto r = layouts[0], tp = layouts[2];
  std::vector<RegionSummary> regions(3);
  regions[0].interface = {{{0, type}}, {{1, type}}};
  regions[1].interface = {{{1, type}}, {{2, type}}};
  regions[2].interface = {{{1, type}, {2, type}}, {{3, type}}};
  regions[0].plans = {{0, {{r}, {tp}}, {1, 0, 0}, ""},
                      {1, {{r}, {r}}, {3, 0, 0}, ""}};
  regions[1].plans = {{0, {{r}, {r}}, {1, 0, 0}, ""},
                      {1, {{tp}, {r}}, {1, 0, 0}, ""}};
  regions[2].plans = {{0, {{r, r}, {r}}, {1, 0, 0}, ""},
                      {1, {{tp, r}, {r}}, {4, 0, 0}, ""}};
  DagInterface graph;
  graph.external = {{{0, type}}, {{3, type}}};
  for (size_t i = 0; i < regions.size(); ++i) {
    regions[i].id = i;
    regions[i].finalizePlans();
    graph.regions.push_back(regions[i].interface);
  }
  graph.live_after = buildLiveCuts(graph.external, graph.regions);
  ReshardPlanner oracle = [&](auto from, auto to, auto tensor) {
    return ReshardPlan{true, {0, from == to ? 0.0 : 10.0, 0}, from, to, tensor,
                       {}};
  };
  BoundaryState contract{{r}, {r}};
  auto exact = optimizeDag(regions, graph, contract, oracle);
  auto greedy =
      optimizeDag(regions, graph, contract, oracle, {DagSearchMode::Greedy});
  ASSERT_TRUE(exact.feasible && greedy.feasible);
  EXPECT_EQ(exact.cost.total(), 5);
  EXPECT_EQ(greedy.cost.total(), 6);
  EXPECT_EQ(exact.cost.total(), exhaustive(regions, graph, contract, oracle));
  DagSearchOptions cap;
  cap.max_states = 1;
  auto bounded = optimizeDag(regions, graph, contract, oracle, cap);
  EXPECT_TRUE(bounded.feasible);
  EXPECT_TRUE(bounded.truncated);
  EXPECT_GT(bounded.states_discarded, 0);
  EXPECT_EQ(bounded.steps.size(), 3);
  cap.max_states = 4096;
  cap.max_transitions = 1;
  bounded = optimizeDag(regions, graph, contract, oracle, cap);
  EXPECT_TRUE(bounded.feasible);
  EXPECT_TRUE(bounded.truncated);
  EXPECT_EQ(bounded.transitions, 3);
  cap.max_live_values = 1;
  EXPECT_THROW(optimizeDag(regions, graph, contract, oracle, cap),
               std::invalid_argument);
  cap.max_live_values = 4;
  cap.max_states = 0;
  EXPECT_THROW(optimizeDag(regions, graph, contract, oracle, cap),
               std::invalid_argument);
  auto unknown = optimizeDag(
      regions, graph, contract, [](auto from, auto to, auto tensor) {
        return ReshardPlan{false, {0, 0, 1}, from, to, tensor, {}};
      });
  EXPECT_FALSE(unknown.feasible);
  EXPECT_TRUE(unknown.execution.nodes.empty());
  auto invalid = graph;
  invalid.live_after[1].pop_back();
  EXPECT_THROW(optimizeDag(regions, invalid, contract, oracle),
               std::invalid_argument);
}

TEST_F(DagOptimizationTest, ConsumerConversionPreservesProducerForLaterUse) {
  auto module = fixture("fanout_join");
  RegionOptimizerOptions options;
  options.optimize_dag = true;
  auto report = summarizeRegions(*module, options);
  auto graph = report.dag->interface;
  auto type = llvm::cast<mlir::RankedTensorType>(graph.external.inputs[0].type);
  auto layouts = tensorLayoutChoices(type, report.mesh, {}, {});
  auto r = layouts[0], tp = layouts[2];
  // Force producer TP, first consumer R, and later consumer TP. The first
  // consumer's all-gather must not overwrite the producer's persistent TP copy.
  std::vector<BoundaryState> boundaries{
      {{r}, {tp}}, {{r, r}, {r}}, {{tp}, {r}}, {{r, r}, {r}}};
  for (size_t i = 0; i < report.regions.size(); ++i) {
    auto& region = report.regions[i];
    auto found = std::find_if(
        region.plans.begin(), region.plans.end(),
        [&](const auto& plan) { return plan.boundary == boundaries[i]; });
    ASSERT_NE(found, region.plans.end());
    auto selected = *found;
    region.plans = {std::move(selected)};
    region.finalizePlans();
  }
  ReshardCostOracle oracle(report.mesh);
  auto result = optimizeDag(report.regions, graph, report.dag->contract,
                            [&](auto from, auto to, auto tensor) {
                              return oracle.plan(from, to, tensor);
                            });
  ASSERT_TRUE(result.feasible);
  EXPECT_EQ(result.steps[1].incoming[0].from, tp);
  EXPECT_EQ(result.steps[1].incoming[0].to, r);
  EXPECT_GT(result.steps[1].incoming[0].cost.communication, 0);
  EXPECT_EQ(result.steps[1].live_layouts[0], tp);
  EXPECT_EQ(result.steps[2].incoming[0].from, tp);
  EXPECT_EQ(result.steps[2].incoming[0].to, tp);
  verifyCostAndOccurrences(result, report, CostModel{});
  auto emitted = parse(materializeExecutionPlan(
      result.execution, report.regions, report.mesh, CostModel{}));
  size_t tanhCount = 0;
  emitted->walk([&](mlir::Operation* op) {
    if (op->getName().getStringRef() == "stablehlo.tanh") ++tanhCount;
  });
  EXPECT_EQ(tanhCount, 2);
  std::mt19937 random(42);
  auto arguments = inputs(*module, random);
  compare(test::interpretFixture(*module, arguments),
          test::interpretFixture(*emitted, arguments));
}

TEST_F(DagOptimizationTest, NumericalSourceRewritesAndEmittedDAGAgree) {
  std::mt19937 random(42);
  for (const auto& name : {"residual_block", "fanout_join",
                           "independent_branches", "dag_multi_output"}) {
    for (bool resolved : {false, true}) {
      SCOPED_TRACE(std::string(name) + " resolved=" + std::to_string(resolved));
      auto module = fixture(name);
      RegionOptimizerOptions options;
      options.optimize_dag = true;
      options.dag.mode =
          resolved ? DagSearchMode::Resolved : DagSearchMode::Exact;
      options.cost.compute_work_per_us = 1;
      options.cost.collective_latency_us = 0.01;
      auto report = summarizeRegions(*module, options);
      const auto& dag = *report.dag;
      verifyCostAndOccurrences(dag.selected(), report, CostModel(options.cost));
      auto emitted = parse(dag.selected().lowered_mlir);
      auto sourceRegions =
          Regionizer().split(module->lookupSymbol<mlir::func::FuncOp>("main"));
      for (size_t sample = 0; sample < 6; ++sample) {
        auto arguments = inputs(*module, random, sample == 5);
        auto expected = test::interpretFixture(*module, arguments);
        compare(expected, test::interpretFixture(*emitted, arguments));
        std::map<ValueId, test::DenseTensor> values;
        for (size_t i = 0; i < arguments.size(); ++i)
          values.emplace(dag.interface.external.inputs[i].value, arguments[i]);
        for (const auto& step : dag.selected().steps) {
          const auto& summary = report.regions[step.region];
          auto candidate = prepareCandidateModule(
              sourceRegions[step.region],
              summary.candidates.at(step.selected.candidate_id), report.mesh);
          std::vector<test::DenseTensor> regionInputs;
          for (auto port : summary.interface.inputs)
            regionInputs.push_back(values.at(port.value));
          auto outputs = test::interpretFixture(*candidate, regionInputs);
          for (size_t j = 0; j < outputs.size(); ++j)
            values.emplace(summary.interface.outputs[j].value, outputs[j]);
        }
        std::vector<test::DenseTensor> rewritten;
        for (auto port : dag.interface.external.outputs)
          rewritten.push_back(values.at(port.value));
        compare(expected, rewritten);
      }
    }
  }
}

TEST_F(DagOptimizationTest,
       RepeatedOperandHasOnePortAndEarlyReturnsRetainOrder) {
  auto module = parse(R"mlir(module {sdy.mesh @mesh = <["data"=2, "model"=2]>
    func.func @main(%x: tensor<8x4xf32>) -> (tensor<8x4xf32>, tensor<8x4xf32>, tensor<8x4xf32>) {
      %a = stablehlo.tanh %x : tensor<8x4xf32>
      %b = stablehlo.add %a, %a : tensor<8x4xf32>
      %c = stablehlo.tanh %b : tensor<8x4xf32>
      return %c, %a, %c : tensor<8x4xf32>, tensor<8x4xf32>, tensor<8x4xf32>
    }})mlir");
  RegionOptimizerOptions options;
  options.optimize_dag = true;
  auto report = summarizeRegions(*module, options);
  const auto& dag = *report.dag;
  ASSERT_EQ(dag.interface.regions[1].inputs.size(), 1);
  EXPECT_EQ(dag.selected().steps[1].incoming.size(), 1);
  EXPECT_EQ(ids(dag.interface.live_after[2]), (std::vector<ValueId>{1, 3}));
  // An early return pins the produced layout, not every consumer's input.
  bool consumerCanShard = false;
  for (const auto& plan : report.regions[1].plans)
    consumerCanShard |= plan.boundary.inputs[0] != dag.contract.outputs[1];
  EXPECT_TRUE(consumerCanShard);
  auto emitted = parse(dag.selected().lowered_mlir);
  auto main = emitted->lookupSymbol<mlir::func::FuncOp>("main");
  auto ret =
      llvm::cast<mlir::func::ReturnOp>(main.getBody().front().getTerminator());
  EXPECT_EQ(ret.getOperand(0), ret.getOperand(2));
  EXPECT_NE(ret.getOperand(0), ret.getOperand(1));
}

TEST_F(DagOptimizationTest, RejectsIncompleteCoverageAndInvalidConfiguration) {
  auto module = fixture("residual_block");
  RegionOptimizerOptions options;
  options.optimize_dag = true;
  options.dag.max_live_values = 1;
  EXPECT_THROW(summarizeRegions(*module, options), std::invalid_argument);
  options.dag.max_live_values = 4;
  options.dag.mode = DagSearchMode::Greedy;
  EXPECT_THROW(summarizeRegions(*module, options), std::invalid_argument);
  options.dag.mode = DagSearchMode::Exact;
  options.optimize_chain = true;
  EXPECT_THROW(summarizeRegions(*module, options), std::invalid_argument);
  auto blocked = parse(R"mlir(module {sdy.mesh @mesh = <["data"=2, "model"=2]>
    func.func @main(%x: tensor<8x4xf32>) -> tensor<8x4xf32> {
      %a = stablehlo.tanh %x : tensor<8x4xf32>
      %b = sdy.sharding_constraint %a <@mesh, [{}, {}]> : tensor<8x4xf32>
      %c = stablehlo.tanh %b : tensor<8x4xf32>
      return %c : tensor<8x4xf32>
    }})mlir");
  options.optimize_chain = false;
  EXPECT_THROW(summarizeRegions(*blocked, options), std::invalid_argument);
  auto main = module->lookupSymbol<mlir::func::FuncOp>("main");
  auto regions = Regionizer().split(main);
  std::swap(regions[0], regions[1]);
  EXPECT_THROW(buildDagInterface(main, regions, ValueIndex(*module)),
               std::invalid_argument);
}

TEST_F(DagOptimizationTest, FixedAnnotationsSharedArgumentsAndEarlyReturns) {
  auto module = parse(R"mlir(module {
    sdy.mesh @mesh = <["data"=2, "model"=2]>
    func.func @main(%unused: tensor<8x4xf32>, %x: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {?}]>}, %w: tensor<8x4xf32>) -> (tensor<8x4xf32>, tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {}]>}, tensor<8x4xf32>) {
      %a = stablehlo.tanh %x : tensor<8x4xf32>
      %b = stablehlo.add %a, %w : tensor<8x4xf32>
      %c = stablehlo.tanh %a : tensor<8x4xf32>
      %y = stablehlo.add %b, %c : tensor<8x4xf32>
      return %y, %a, %y : tensor<8x4xf32>, tensor<8x4xf32>, tensor<8x4xf32>
    }
  })mlir");
  RegionOptimizerOptions options;
  options.optimize_dag = true;
  auto report = summarizeRegions(*module, options);
  const auto& dag = *report.dag;
  EXPECT_TRUE(dag.contract.inputs[1].attr.isFullyClosed());
  EXPECT_EQ(dag.contract.inputs[1], dag.contract.outputs[1]);
  EXPECT_EQ(dag.exact.steps[0].selected.boundary.outputs[0],
            dag.contract.outputs[1]);
  EXPECT_EQ(dag.exact.peak_live_values, 3);
  auto emitted = parse(dag.selected().lowered_mlir);
  auto main = emitted->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_EQ(main.getNumArguments(), 3);
  EXPECT_EQ(mlir::sdy::getSharding(main.getArgument(1)),
            dag.contract.inputs[1].attr);
  auto ret =
      llvm::cast<mlir::func::ReturnOp>(main.getBody().front().getTerminator());
  EXPECT_EQ(ret.getOperand(0), ret.getOperand(2));
  EXPECT_EQ(mlir::sdy::getSharding(ret.getOperand(1)),
            dag.contract.outputs[1].attr);
  auto reference = parse(print(*module));
  auto referenceMain = reference->lookupSymbol<mlir::func::FuncOp>("main");
  referenceMain.removeArgAttr(1, "sdy.sharding");
  referenceMain.removeResultAttr(1, "sdy.sharding");
  std::mt19937 random(42);
  auto arguments = inputs(*reference, random);
  compare(test::interpretFixture(*reference, arguments),
          test::interpretFixture(*emitted, arguments));
}

TEST_F(DagOptimizationTest, StrictPolicyAndWriterEmitSelectedDAG) {
  auto module = fixture("residual_block");
  RegionOptimizerOptions options;
  options.optimize_dag = true;
  options.dag.mode = DagSearchMode::Resolved;
  options.rewriting.numerical_policy = NumericalPolicy::PreserveEvaluation;
  llvm::SmallString<128> directory;
  ASSERT_FALSE(
      llvm::sys::fs::createUniqueDirectory("joint-dag-test", directory));
  ArtifactWriter writer(directory.str().str());
  auto report = summarizeRegions(*module, options, writer.observer());
  auto path = directory;
  llvm::sys::path::append(path, "selected.mlir");
  auto file = llvm::MemoryBuffer::getFile(path);
  ASSERT_TRUE(file);
  EXPECT_EQ((*file)->getBuffer(), report.dag->selected().lowered_mlir);
  auto emitted = parse((*file)->getBuffer());
  std::mt19937 random(42);
  auto arguments = inputs(*module, random);
  compare(test::interpretFixture(*module, arguments),
          test::interpretFixture(*emitted, arguments));
  EXPECT_FALSE(llvm::sys::fs::remove_directories(directory));
}
}  // namespace
}  // namespace joint_shard

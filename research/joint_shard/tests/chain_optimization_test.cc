#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <random>
#include <set>

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
class ChainOptimizationTest : public test::MlirTest {
 protected:
  mlir::OwningOpRef<mlir::ModuleOp> fixture(size_t count) {
    auto module = mlir::parseSourceFile<mlir::ModuleOp>(
        "research/joint_shard/testdata/chain_" + std::to_string(count) +
            ".mlir",
        &context);
    if (!module) throw std::runtime_error("missing chain fixture");
    return module;
  }
  // Independent Cartesian enumeration: no prefix merging or plan resolution.
  double exhaustive(const std::vector<RegionSummary>& regions,
                    const ChainInterface& interface,
                    const BoundaryState& contract,
                    const ReshardPlanner& oracle) {
    double best = std::numeric_limits<double>::infinity();
    std::vector<const RegionPlan*> selected;
    std::function<void(size_t)> visit = [&](size_t index) {
      if (index == regions.size()) {
        Cost cost;
        for (size_t i = 0; i < selected.size(); ++i) {
          const auto& p = *selected[i];
          for (size_t j = 0; j < interface.inputs[i].size(); ++j)
            if (interface.inputs[i][j] >= 0 &&
                p.boundary.inputs[j] != contract.inputs[interface.inputs[i][j]])
              return;
          cost += p.cost;
          if (i) {
            auto adapter =
                oracle(selected[i - 1]->boundary.outputs[0],
                       p.boundary.inputs[interface.intermediate_inputs[i]],
                       regions[i - 1].interface.outputs[0].type);
            if (!adapter.feasible) return;
            cost += adapter.cost;
          }
        }
        for (size_t i = 0; i < interface.external.outputs.size(); ++i) {
          auto found = std::find(regions.back().interface.outputs.begin(),
                                 regions.back().interface.outputs.end(),
                                 interface.external.outputs[i]);
          if (selected.back()
                  ->boundary
                  .outputs[found - regions.back().interface.outputs.begin()] !=
              contract.outputs[i])
            return;
        }
        if (cost.known()) best = std::min(best, cost.total());
        return;
      }
      for (const auto& plan : regions[index].plans) {
        selected.push_back(&plan);
        visit(index + 1);
        selected.pop_back();
      }
    };
    visit(0);
    return best;
  }
  void compare(const std::vector<test::DenseTensor>& expected,
               const std::vector<test::DenseTensor>& actual) {
    ASSERT_EQ(expected.size(), actual.size());
    // Explicit relaxed f32 policy for bounded inputs, including cancellation.
    // Distributed reductions may change accumulation order even with strict
    // rewriting. No bitwise-equivalence claim is made.
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
};

TEST_F(ChainOptimizationTest, ThreeToFiveRegionsMatchExhaustiveAndEmitOneMain) {
  for (size_t n : {3, 4, 5}) {
    SCOPED_TRACE(n);
    auto module = fixture(n);
    auto before = print(*module);
    RegionOptimizerOptions options;
    options.optimize_chain = true;
    auto report = summarizeRegions(*module, options);
    ASSERT_TRUE(report.chain);
    const auto& chain = *report.chain;
    ASSERT_EQ(report.regions.size(), n);
    ASSERT_EQ(chain.exact.steps.size(), n);
    EXPECT_EQ(print(*module), before);
    EXPECT_FALSE(chain.exact.truncated);
    EXPECT_FALSE(chain.boundary_truncated);
    ReshardCostOracle oracle(report.mesh);
    ReshardPlanner planner = [&](auto from, auto to, auto type) {
      return oracle.plan(from, to, type);
    };
    EXPECT_NEAR(
        chain.exact.cost.total(),
        exhaustive(report.regions, chain.interface, chain.contract, planner),
        1e-9);
    EXPECT_LE(chain.resolved.cost.total(), chain.exact.cost.total() + 1e-9);
    EXPECT_LE(chain.exact.cost.total(), chain.original.cost.total() + 1e-9);
    EXPECT_LT(chain.exact.cost.total(), chain.greedy.cost.total());
    auto selected = parse(chain.selected().lowered_mlir);
    EXPECT_EQ(std::distance(selected->getOps<mlir::func::FuncOp>().begin(),
                            selected->getOps<mlir::func::FuncOp>().end()),
              1);
    auto main = selected->lookupSymbol<mlir::func::FuncOp>("main");
    for (size_t i = 0; i < main.getNumArguments(); ++i)
      EXPECT_EQ(mlir::sdy::getSharding(main.getArgument(i)),
                chain.contract.inputs[i].attr);
    auto actual = CostModel().estimate(*selected, report.mesh.mesh);
    EXPECT_NEAR(actual.compute, chain.selected().cost.compute, 1e-9);
    EXPECT_NEAR(actual.communication, chain.selected().cost.communication,
                1e-9);
    auto text = formatReport(chain);
    EXPECT_NE(text.find("original + DP"), std::string::npos);
    EXPECT_NE(text.find("Savings versus greedy"), std::string::npos);
    // Real lowered greedy winner has an executable nonidentity seam adapter.
    auto greedy = parse(materializeExecutionPlan(
        chain.greedy.execution, report.regions, report.mesh, CostModel{}));
    EXPECT_NEAR(CostModel().estimate(*greedy, report.mesh.mesh).total(),
                chain.greedy.cost.total(), 1e-9);
    auto original = parse(materializeExecutionPlan(chain.original.execution,
                                                   report.original_regions,
                                                   report.mesh, CostModel{}));
    EXPECT_NEAR(CostModel().estimate(*original, report.mesh.mesh).total(),
                chain.original.cost.total(), 1e-9);
  }
}

TEST_F(ChainOptimizationTest, SyntheticDPBeatsGreedyForEveryChainLength) {
  auto module = fixture(3);
  auto mesh = selectMesh(*module);
  auto type =
      mlir::RankedTensorType::get({8, 4}, mlir::Float32Type::get(&context));
  auto choices = tensorLayoutChoices(type, mesh, {}, {});
  auto r = choices[0], tp = choices[2];
  ReshardPlanner oracle = [&](auto from, auto to, auto tensor) {
    return ReshardPlan{true, {0, from == to ? 0.0 : 10.0, 0}, from, to, tensor,
                       {}};
  };
  for (size_t n : {3, 4, 5}) {
    ChainInterface interface;
    interface.external = {{{0, type}}, {{n, type}}};
    std::vector<RegionSummary> regions(n);
    for (size_t i = 0; i < n; ++i) {
      auto& region = regions[i];
      region.id = i;
      region.interface = {{{i, type}}, {{i + 1, type}}};
      region.plans = {{0, {{r}, {r}}, {3, 0, 0}, ""}};
      if (!i) region.plans.push_back({1, {{r}, {tp}}, {1, 0, 0}, ""});
      region.finalizePlans();
      interface.regions.push_back(region.interface);
      interface.inputs.push_back({i ? -1 : 0});
      interface.intermediate_inputs.push_back(0);
    }
    BoundaryState contract{{r}, {r}};
    auto dp = optimizeChain(regions, interface, contract, oracle);
    auto greedy = optimizeChain(regions, interface, contract, oracle,
                                {ChainSearchMode::Greedy});
    ASSERT_TRUE(dp.feasible && greedy.feasible);
    EXPECT_EQ(dp.cost.total(), 3 * n);
    EXPECT_LT(dp.cost.total(), greedy.cost.total());
    EXPECT_EQ(dp.cost.total(),
              exhaustive(regions, interface, contract, oracle));
    auto repeat = optimizeChain(regions, interface, contract, oracle);
    EXPECT_EQ(dp.steps[0].selected.requested,
              repeat.steps[0].selected.requested);
    auto capped = optimizeChain(regions, interface, contract, oracle,
                                {ChainSearchMode::Exact, 1});
    EXPECT_TRUE(capped.truncated);
    EXPECT_EQ(capped.steps.size(), n);
    EXPECT_THROW(optimizeChain(regions, interface, contract, oracle,
                               {ChainSearchMode::Exact, 0}),
                 std::invalid_argument);
    auto unknown = optimizeChain(
        regions, interface, contract, [](auto from, auto to, auto tensor) {
          return ReshardPlan{false, {0, 0, 1}, from, to, tensor, {}};
        });
    EXPECT_FALSE(unknown.feasible);
  }
}

TEST_F(ChainOptimizationTest,
       ContractsSharedArgumentsUnusedArgumentsAndReturnOrder) {
  auto module = parse(R"mlir(module {
    sdy.mesh @mesh = <["data"=2, "model"=2]>
    func.func @main(%unused: tensor<8x4xf32>, %x: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {?}]>}, %y: tensor<8x4xf32>) -> (tensor<8x4xf32>, tensor<8x4xf32>, tensor<8x4xf32>) {
      %a = stablehlo.add %y, %x : tensor<8x4xf32>
      %b = stablehlo.tanh %a : tensor<8x4xf32>
      %c = stablehlo.add %b, %x : tensor<8x4xf32>
      %d = stablehlo.multiply %c, %y : tensor<8x4xf32>
      return %d, %c, %d : tensor<8x4xf32>, tensor<8x4xf32>, tensor<8x4xf32>
    }
  })mlir");
  RegionOptimizerOptions options;
  options.optimize_chain = true;
  options.max_boundary_states = 9;
  auto report = summarizeRegions(*module, options);
  ASSERT_EQ(report.regions.size(), 3);
  const auto& chain = *report.chain;
  EXPECT_EQ(chain.interface.inputs[0], (std::vector<int64_t>{2, 1}));
  EXPECT_EQ(chain.interface.inputs[2], (std::vector<int64_t>{-1, 1, 2}));
  EXPECT_EQ(
      chain.contract.inputs[1].attr.getDimSharding(0).getAxes()[0].getName(),
      "data");
  EXPECT_TRUE(chain.contract.inputs[1].attr.isFullyClosed());
  EXPECT_EQ(chain.contract.inputs[0],
            replicatedSharding(llvm::cast<mlir::RankedTensorType>(
                                   chain.interface.external.inputs[0].type),
                               report.mesh));
  EXPECT_EQ(chain.interface.external.outputs[0],
            chain.interface.external.outputs[2]);
  EXPECT_FALSE(chain.boundary_truncated);
  auto result = parse(chain.selected().lowered_mlir);
  auto main = result->lookupSymbol<mlir::func::FuncOp>("main");
  EXPECT_EQ(main.getNumArguments(), 3);
  auto ret =
      llvm::cast<mlir::func::ReturnOp>(main.getBody().front().getTerminator());
  EXPECT_EQ(ret.getOperand(0), ret.getOperand(2));
  EXPECT_NE(ret.getOperand(0), ret.getOperand(1));
  auto reference = parse(print(*module));
  auto referenceMain = reference->lookupSymbol<mlir::func::FuncOp>("main");
  referenceMain.removeArgAttr(1, "sdy.sharding");
  std::vector<test::DenseTensor> inputs{test::DenseTensor(32, 0),
                                        test::DenseTensor(32, 0.3f),
                                        test::DenseTensor(32, -0.2f)};
  compare(test::interpretFixture(*reference, inputs),
          test::interpretFixture(*result, inputs));
}

TEST_F(ChainOptimizationTest, RejectsFanoutResidualsAndPreservedOperations) {
  RegionOptimizerOptions options;
  options.optimize_chain = true;
  for (
      auto source : {
          R"mlir(module {func.func @main(%x: tensor<8x4xf32>) -> (tensor<8x4xf32>, tensor<8x4xf32>) {
      %a = stablehlo.tanh %x : tensor<8x4xf32>
      %b = stablehlo.tanh %a : tensor<8x4xf32>
      return %a, %b : tensor<8x4xf32>, tensor<8x4xf32> }})mlir",
          R"mlir(module {func.func @main(%x: tensor<8x4xf32>) -> tensor<8x4xf32> {
      %a = stablehlo.tanh %x : tensor<8x4xf32>
      %b = stablehlo.tanh %a : tensor<8x4xf32>
      %c = stablehlo.add %b, %a : tensor<8x4xf32>
      return %c : tensor<8x4xf32> }})mlir",
          R"mlir(module {sdy.mesh @mesh = <["data"=2, "model"=2]>
      func.func @main(%x: tensor<8x4xf32>) -> tensor<8x4xf32> {
      %a = stablehlo.tanh %x : tensor<8x4xf32>
      %b = sdy.sharding_constraint %a <@mesh, [{}, {}]> : tensor<8x4xf32>
      %c = stablehlo.tanh %b : tensor<8x4xf32>
      return %c : tensor<8x4xf32> }})mlir"}) {
    auto module = parse(source);
    EXPECT_THROW(summarizeRegions(*module, options), std::invalid_argument);
  }
}

TEST_F(ChainOptimizationTest,
       ExecutesSelectedRewritesAndEmittedCollectivesNumerically) {
  std::mt19937 random(42);
  std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
  for (size_t n : {3, 4, 5}) {
    for (bool resolved : {false, true}) {
      SCOPED_TRACE(std::to_string(n) + " resolved=" + std::to_string(resolved));
      auto module = fixture(n);
      RegionOptimizerOptions options;
      options.optimize_chain = true;
      options.chain_resolved = resolved;
      options.cost.compute_work_per_us = 1;
      options.cost.collective_latency_us = 0.01;
      auto report = summarizeRegions(*module, options);
      const auto& chain = *report.chain;
      auto selected = parse(chain.selected().lowered_mlir);
      EXPECT_NE(chain.selected().lowered_mlir.find("sdy.all_gather"),
                std::string::npos);
      // The artifact count equals leaf execution occurrences, including child
      // collectives and every seam/wrapper adapter. Composite references recur
      // once in this chain, so each leaf is emitted once.
      size_t operations = 0;
      for (const auto& node : chain.selected().execution.nodes) {
        std::string artifact;
        if (node.kind == PlanKind::Region)
          artifact = report.regions[node.region]
                         .plans[node.implementation]
                         .lowered_mlir;
        if (node.kind == PlanKind::Adapter)
          artifact = node.adapter.lowered_mlir;
        if (artifact.empty()) continue;
        auto leaf = parse(artifact);
        operations +=
            std::distance(leaf->lookupSymbol<mlir::func::FuncOp>("main")
                              .getBody()
                              .front()
                              .without_terminator()
                              .begin(),
                          leaf->lookupSymbol<mlir::func::FuncOp>("main")
                              .getBody()
                              .front()
                              .without_terminator()
                              .end());
      }
      auto emitted = selected->lookupSymbol<mlir::func::FuncOp>("main");
      EXPECT_EQ(
          std::distance(emitted.getBody().front().without_terminator().begin(),
                        emitted.getBody().front().without_terminator().end()),
          operations);
      auto actualCost =
          CostModel(options.cost).estimate(*selected, report.mesh.mesh);
      EXPECT_NEAR(actualCost.compute, chain.selected().cost.compute, 1e-9);
      EXPECT_NEAR(actualCost.communication, chain.selected().cost.communication,
                  1e-9);
      auto sourceRegions =
          Regionizer(options.regionizer)
              .split(module->lookupSymbol<mlir::func::FuncOp>("main"));
      for (size_t sample = 0; sample < 6; ++sample) {
        std::vector<test::DenseTensor> inputs{{0.3f},
                                              test::DenseTensor(128),
                                              test::DenseTensor(64),
                                              test::DenseTensor(16)};
        for (size_t i = 1; i < inputs.size(); ++i)
          for (size_t j = 0; j < inputs[i].size(); ++j)
            inputs[i][j] =
                sample == 5 ? (j % 2 ? -0.5f : 0.5f) : distribution(random);
        auto expected = test::interpretFixture(*module, inputs);
        compare(expected, test::interpretFixture(*selected, inputs));
        std::map<ValueId, test::DenseTensor> values;
        for (size_t i = 0; i < inputs.size(); ++i)
          values[chain.interface.external.inputs[i].value] = inputs[i];
        for (const auto& step : chain.selected().steps) {
          const auto& summary = report.regions[step.region];
          auto candidate = prepareCandidateModule(
              sourceRegions[step.region],
              summary.candidates.at(step.selected.candidate_id), report.mesh);
          std::vector<test::DenseTensor> arguments;
          for (auto port : summary.interface.inputs)
            arguments.push_back(values.at(port.value));
          auto outputs = test::interpretFixture(*candidate, arguments);
          for (size_t j = 0; j < outputs.size(); ++j)
            values[summary.interface.outputs[j].value] = outputs[j];
        }
        std::vector<test::DenseTensor> rewritten;
        for (auto port : chain.interface.external.outputs)
          rewritten.push_back(values.at(port.value));
        compare(expected, rewritten);
      }
    }
  }
}

TEST_F(ChainOptimizationTest,
       EveryCanonicalAdapterExecutesAndReductionsCombineDevices) {
  auto module = fixture(3);
  auto mesh = selectMesh(*module);
  auto type =
      mlir::RankedTensorType::get({8, 4}, mlir::Float32Type::get(&context));
  auto layouts = tensorLayoutChoices(type, mesh, {}, {});
  ReshardCostOracle oracle(mesh);
  test::DenseTensor input(32);
  for (size_t i = 0; i < input.size(); ++i) input[i] = float(i) / 32 - 0.5f;
  for (auto from : layouts)
    for (auto to : layouts) {
      auto adapter = oracle.plan(from, to, type);
      ASSERT_TRUE(adapter.feasible);
      if (from == to) {
        EXPECT_EQ(adapter.cost.total(), 0);
        continue;
      }
      auto artifact = parse(adapter.lowered_mlir);
      compare({input}, test::interpretFixture(*artifact, {input}));
    }
  auto dot = parse(R"mlir(module {
    sdy.mesh @mesh = <["data"=2, "model"=2]>
    func.func @main(%x: tensor<8x16xf32>, %w: tensor<16x4xf32>) -> tensor<8x4xf32> {
      %y = stablehlo.dot_general %x, %w, contracting_dims = [1] x [0] : (tensor<8x16xf32>, tensor<16x4xf32>) -> tensor<8x4xf32>
      return %y : tensor<8x4xf32>
    }
  })mlir");
  auto function = dot->lookupSymbol<mlir::func::FuncOp>("main");
  auto lhsType =
      llvm::cast<mlir::RankedTensorType>(function.getArgument(0).getType());
  auto rhsType =
      llvm::cast<mlir::RankedTensorType>(function.getArgument(1).getType());
  LayoutPolicy policy;
  auto lhs = tensorLayoutChoices(lhsType, mesh, policy, {0, 1});
  auto rhs = tensorLayoutChoices(rhsType, mesh, policy, {1, 0});
  BoundaryState boundary{{lhs[2], rhs[2]}, {layouts[0]}};
  auto evaluated = RegionEvaluator(mesh).evaluate(*dot, boundary);
  ASSERT_TRUE(evaluated.feasible);
  ASSERT_NE(evaluated.lowered_mlir.find("sdy.all_reduce"), std::string::npos);
  std::vector<test::DenseTensor> arguments{test::DenseTensor(128),
                                           test::DenseTensor(64)};
  for (size_t i = 0; i < arguments[0].size(); ++i)
    arguments[0][i] = float(int(i % 17) - 8) / 16;
  for (size_t i = 0; i < arguments[1].size(); ++i)
    arguments[1][i] = float(int(i % 13) - 6) / 16;
  auto expected = test::interpretFixture(*dot, arguments);
  auto lowered = parse(evaluated.lowered_mlir);
  compare(expected, test::interpretFixture(*lowered, arguments));
  // Prove the simulator does not silently treat a reduction as identity.
  lowered->walk([&](mlir::sdy::AllReduceOp reduce) {
    reduce.getResult().replaceAllUsesWith(reduce.getTensor());
    reduce.erase();
  });
  EXPECT_THROW(test::interpretFixture(*lowered, arguments),
               std::invalid_argument);
}

TEST_F(ChainOptimizationTest,
       StrictPolicyAndArtifactWriterUseSelectedContract) {
  auto module = fixture(3);
  RegionOptimizerOptions options;
  options.optimize_chain = true;
  options.chain_resolved = true;
  options.rewriting.numerical_policy = NumericalPolicy::PreserveEvaluation;
  llvm::SmallString<128> directory;
  ASSERT_FALSE(
      llvm::sys::fs::createUniqueDirectory("joint-chain-test", directory));
  ArtifactWriter writer(directory.str().str());
  auto report = summarizeRegions(*module, options, writer.observer());
  auto path = directory;
  llvm::sys::path::append(path, "selected.mlir");
  auto file = llvm::MemoryBuffer::getFile(path);
  ASSERT_TRUE(file);
  EXPECT_EQ((*file)->getBuffer(), report.chain->selected().lowered_mlir);
  auto artifact = parse((*file)->getBuffer());
  std::vector<test::DenseTensor> inputs{{0.3f},
                                        test::DenseTensor(128, 0.2f),
                                        test::DenseTensor(64, -0.1f),
                                        test::DenseTensor(16, 0.4f)};
  compare(test::interpretFixture(*module, inputs),
          test::interpretFixture(*artifact, inputs));
  for (const auto& step : report.chain->selected().steps)
    EXPECT_EQ(step.selected.candidate_id, 0);
  EXPECT_FALSE(llvm::sys::fs::remove_directories(directory));
}

TEST_F(ChainOptimizationTest, ReportsEverySearchBudgetAndKeepsFinalContract) {
  auto module = fixture(5);
  RegionOptimizerOptions options;
  options.optimize_chain = true;
  options.max_candidates = 1;
  options.max_boundary_states = 1;
  auto report = summarizeRegions(*module, options);
  ASSERT_TRUE(report.chain->selected().feasible);
  EXPECT_TRUE(report.chain->boundary_truncated);
  EXPECT_TRUE(report.chain->candidate_cap_reached);
  EXPECT_EQ(report.chain->selected().contract, report.chain->contract);
  EXPECT_EQ(report.chain->selected().steps.size(), 5);
  EXPECT_NEAR(report.chain->exact.cost.total(),
              report.chain->original.cost.total(), 1e-9);
  auto main = parse(report.chain->selected().lowered_mlir);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*main)));
  auto text = formatReport(*report.chain);
  EXPECT_NE(text.find("Boundary truncated: 1"), std::string::npos);
  EXPECT_NE(text.find("profiles skipped at cap: 1"), std::string::npos);
}
}  // namespace
}  // namespace joint_shard

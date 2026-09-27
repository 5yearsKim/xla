#include <map>
#include <set>

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"
#include "research/joint_shard/search/region_optimizer.h"
#include "shardy/dialect/sdy/ir/register.h"
#include "stablehlo/dialect/Register.h"

namespace {
class PairCompositionTest : public ::testing::Test {
 protected:
  mlir::MLIRContext context;
  PairCompositionTest() {
    mlir::DialectRegistry registry;
    mlir::stablehlo::registerAllDialects(registry);
    mlir::sdy::registerAllDialects(registry);
    context.appendDialectRegistry(registry);
  }
  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef source) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
    if (!module || mlir::failed(mlir::verify(*module)))
      throw std::runtime_error("invalid test fixture");
    return module;
  }
  std::string print(mlir::ModuleOp module) {
    std::string result;
    llvm::raw_string_ostream out(result);
    module.print(out);
    return result;
  }
  mlir::OwningOpRef<mlir::ModuleOp> chain() {
    auto module = mlir::parseSourceFile<mlir::ModuleOp>(
        "research/joint_shard/testdata/pair_scaling_dot.mlir", &context);
    if (!module) throw std::runtime_error("missing pair fixture");
    return module;
  }
};
TEST_F(PairCompositionTest, JointSearchBeatsGreedyAndPreservesOuterState) {
  auto module = chain();
  auto mesh = selectMesh(*module);
  auto type =
      mlir::RankedTensorType::get({8, 4}, mlir::Float32Type::get(&context));
  auto layouts = tensorLayoutChoices(type, mesh, {}, {});
  auto r = layouts[0], tp = layouts[2];
  RegionSummary a, b;
  a.id = 0;
  b.id = 1;
  a.interface = {{{0, type}}, {{1, type}}};
  b.interface = {{{1, type}}, {{2, type}}};
  a.plans = {{0, {{r}, {tp}}, {1, 0, 0}, ""}, {1, {{r}, {r}}, {3, 0, 0}, ""}};
  b.plans = {{0, {{r}, {r}}, {1, 0, 0}, ""}, {1, {{tp}, {r}}, {4, 0, 0}, ""}};
  a.keepBestPerBoundary();
  b.keepBestPerBoundary();
  PairInterface interface;
  interface.external = {{{0, type}}, {{2, type}}};
  interface.intermediate = {1, type};
  interface.a_inputs = {0};
  interface.b_inputs = {-1};
  ReshardPlanner oracle = [&](auto from, auto to, auto tensor) {
    return ReshardPlan{
        true, {0, from == tp && to == r ? 5.0 : 0.0, 0}, from, to, tensor, {}};
  };
  auto result = composePair(a, b, interface, oracle);
  ASSERT_EQ(result.plans.size(), 1);
  EXPECT_EQ(result.evaluations, 4);
  EXPECT_EQ(result.plans[0].cost.total(), 4);
  EXPECT_EQ(a.plans[result.plans[0].a.requested].candidate_id, 1);
  EXPECT_EQ(b.plans[result.plans[0].b.requested].candidate_id, 0);
  EXPECT_EQ(result.plans[0].boundary, (BoundaryState{{r}, {r}}));
  // The independent local choices cost 1 + TP->R(5) + 1 = 7.
  EXPECT_LT(result.plans[0].cost.total(), 7);
  double exhaustive = 1e100;
  for (const auto& ap : a.plans)
    for (const auto& bp : b.plans)
      exhaustive = std::min(exhaustive, ap.cost.total() +
                                            oracle(ap.boundary.outputs[0],
                                                   bp.boundary.inputs[0], type)
                                                .cost.total() +
                                            bp.cost.total());
  EXPECT_EQ(result.plans[0].cost.total(), exhaustive);
  auto capped = composePair(a, b, interface, oracle, 1);
  EXPECT_TRUE(capped.truncated);
  EXPECT_EQ(capped.evaluations, 1);
  EXPECT_THROW(composePair(a, b, interface, oracle, 0), std::invalid_argument);
  auto unknown =
      composePair(a, b, interface, [](auto from, auto to, auto type) {
        return ReshardPlan{false, {0, 0, 1}, from, to, type, {}};
      });
  EXPECT_TRUE(unknown.plans.empty());
  EXPECT_EQ(unknown.unknown_cost_rejections, 4);
}
TEST_F(PairCompositionTest,
       InterfacesUseSSAIdentityAndDeduplicateSharedInputs) {
  auto module = parse(R"mlir(module {
    func.func @main(%x: tensor<8x4xf32>, %y: tensor<8x4xf32>) -> tensor<8x4xf32> {
      %a=stablehlo.add %x,%y : tensor<8x4xf32>
      %b=stablehlo.multiply %a,%x : tensor<8x4xf32>
      return %b : tensor<8x4xf32>
    }
  })mlir");
  auto fn = module->lookupSymbol<mlir::func::FuncOp>("main");
  auto regions = Regionizer({1, {}}).split(fn);
  ASSERT_EQ(regions.size(), 2);
  ValueIndex values(*module);
  auto interface = buildPairInterface(regions[0], regions[1], values);
  ASSERT_EQ(interface.external.inputs.size(), 2);
  EXPECT_NE(interface.external.inputs[0].value,
            interface.external.inputs[1].value);
  EXPECT_EQ(interface.external.inputs[0].type,
            interface.external.inputs[1].type);
  EXPECT_EQ(interface.b_inputs, (std::vector<int64_t>{-1, 0}));
  EXPECT_EQ(interface.intermediate.value, 2);
  auto mesh = selectMesh(*module);
  auto type = llvm::cast<mlir::RankedTensorType>(interface.intermediate.type);
  auto layouts = tensorLayoutChoices(type, mesh, {}, {});
  auto r = layouts[0], tp = layouts[2];
  RegionSummary a, b;
  a.interface = values.interface(regions[0]);
  b.interface = values.interface(regions[1]);
  a.plans = {{0, {{r, r}, {r}}, {1, 0, 0}, ""}};
  b.plans = {{0, {{r, tp}, {r}}, {1, 0, 0}, ""},
             {1, {{r, r}, {r}}, {2, 0, 0}, ""}};
  a.keepBestPerBoundary();
  b.keepBestPerBoundary();
  ReshardCostOracle oracle(mesh);
  auto composed =
      composePair(a, b, interface, [&](auto from, auto to, auto type) {
        return oracle.plan(from, to, type);
      });
  EXPECT_EQ(composed.shared_layout_rejections, 1);
  ASSERT_EQ(composed.plans.size(), 1);
  EXPECT_EQ(composed.plans[0].cost.total(), 3);
  auto invalid = interface;
  invalid.external.inputs[0].value = 12345;
  EXPECT_THROW(composePair(a, b, invalid,
                           [&](auto from, auto to, auto type) {
                             return oracle.plan(from, to, type);
                           }),
               std::invalid_argument);
  // Adding a source return use of h is a live-out and must be rejected.
  fn.getBody().front().getTerminator()->setOperand(0, regions[0].outputs[0]);
  EXPECT_THROW(buildPairInterface(regions[0], regions[1], values),
               std::invalid_argument);
}
TEST_F(PairCompositionTest, ExactAndResolvedArtifactsMatchExhaustiveCosts) {
  auto module = chain();
  auto before = print(*module);
  RegionOptimizerOptions options;
  options.compose_regions = {{0, 1}};
  auto exact = summarizeRegions(*module, options);
  ASSERT_EQ(exact.regions.size(), 2);
  ASSERT_EQ(exact.compositions.size(), 1);
  const auto& pair = exact.compositions[0];
  EXPECT_EQ(pair.evaluations, 27 * 9);
  EXPECT_FALSE(pair.truncated);
  ASSERT_EQ(pair.plans.size(), 27);
  ReshardCostOracle oracle(exact.mesh);
  std::map<std::string, double> reference;
  for (const auto& ap : exact.regions[0].plans)
    for (const auto& bp : exact.regions[1].plans) {
      BoundaryState outer{ap.boundary.inputs, bp.boundary.outputs};
      double total =
          ap.cost.total() +
          oracle
              .estimate(ap.boundary.outputs[0], bp.boundary.inputs[0],
                        pair.interface.intermediate.type)
              .total() +
          bp.cost.total();
      auto [it, inserted] = reference.emplace(outer.key(), total);
      if (!inserted) it->second = std::min(it->second, total);
    }
  for (const auto& plan : pair.plans) {
    EXPECT_NEAR(plan.cost.total(), reference.at(plan.boundary.key()), 1e-9);
    auto artifact = parse(plan.lowered_mlir);
    auto cost = CostModel().estimate(*artifact, exact.mesh.mesh);
    EXPECT_NEAR(cost.compute, plan.cost.compute, 1e-9);
    EXPECT_NEAR(cost.communication, plan.cost.communication, 1e-9);
    EXPECT_EQ(std::distance(artifact->getOps<mlir::sdy::MeshOp>().begin(),
                            artifact->getOps<mlir::sdy::MeshOp>().end()),
              1);
  }
  // Winners in this chain align h layouts. Force all nine native layout
  // directions to independently exercise executable nonidentity adapters.
  auto planner = [&](auto from, auto to, auto type) {
    return oracle.plan(from, to, type);
  };
  const auto& a = exact.regions[0];
  const auto& b = exact.regions[1];
  size_t forced = 0;
  for (const auto& ap : a.plans) {
    bool replicated_inputs = true;
    for (size_t i = 0; i < ap.boundary.inputs.size(); ++i)
      replicated_inputs &=
          ap.boundary.inputs[i] ==
          replicatedSharding(
              llvm::cast<mlir::RankedTensorType>(a.interface.inputs[i].type),
              exact.mesh);
    if (!replicated_inputs) continue;
    for (const auto& bp : b.plans) {
      if (bp.boundary.outputs[0] !=
          replicatedSharding(
              llvm::cast<mlir::RankedTensorType>(b.interface.outputs[0].type),
              exact.mesh))
        continue;
      ComposedPlan plan;
      plan.a = resolveRegionPlan(a, ap.id, false, planner);
      plan.b = resolveRegionPlan(b, bp.id, false, planner);
      plan.boundary = {ap.boundary.inputs, bp.boundary.outputs};
      plan.intermediate =
          oracle.plan(ap.boundary.outputs[0], bp.boundary.inputs[0],
                      pair.interface.intermediate.type);
      plan.cost = plan.a.cost;
      plan.cost += plan.intermediate.cost;
      plan.cost += plan.b.cost;
      auto artifact = parse(
          materializePair(a, b, pair.interface, plan, exact.mesh, CostModel{}));
      EXPECT_NEAR(CostModel().estimate(*artifact, exact.mesh.mesh).total(),
                  plan.cost.total(), 1e-9);
      ++forced;
    }
  }
  EXPECT_EQ(forced, 9);
  auto tampered = pair.plans[0];
  tampered.cost.compute += 1;
  EXPECT_THROW(
      materializePair(a, b, pair.interface, tampered, exact.mesh, CostModel{}),
      std::runtime_error);
  options.compose_pruned = true;
  auto resolved = summarizeRegions(*module, options);
  ASSERT_EQ(resolved.compositions[0].plans.size(), reference.size());
  bool replaced = false;
  for (const auto& plan : resolved.compositions[0].plans) {
    EXPECT_LE(plan.cost.total(), reference.at(plan.boundary.key()) + 1e-9);
    auto artifact = parse(plan.lowered_mlir);
    auto cost = CostModel().estimate(*artifact, resolved.mesh.mesh);
    EXPECT_NEAR(cost.total(), plan.cost.total(), 1e-9);
    replaced |= plan.a.requested != plan.a.implementation ||
                plan.b.requested != plan.b.implementation;
  }
  EXPECT_TRUE(replaced);
  for (const auto& region : resolved.regions)
    for (const auto& witness : region.dominance)
      EXPECT_NE(std::find(region.frontier.begin(), region.frontier.end(),
                          witness.replacement_id),
                region.frontier.end());
  EXPECT_EQ(print(*module), before);
}
TEST_F(PairCompositionTest, RealRewriteWinnerDependsOnBoundary) {
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(
      "research/joint_shard/testdata/pair_boundary_rewrite.mlir", &context);
  ASSERT_TRUE(module);
  RegionOptimizerOptions options;
  options.compose_regions = {{0, 1}};
  options.layouts.inputs[1] = {std::nullopt, 1};
  options.layouts.inputs[2] = {std::nullopt, 0};
  auto report = summarizeRegions(*module, options);
  const auto& summary = report.regions[0];
  ASSERT_EQ(summary.candidates.size(), 2);
  auto region =
      Regionizer().split(module->lookupSymbol<mlir::func::FuncOp>("main"))[0];
  auto original =
      prepareCandidateModule(region, summary.candidates[0], report.mesh);
  auto factored =
      prepareCandidateModule(region, summary.candidates[1], report.mesh);
  BoundaryState replicated;
  for (auto port : summary.interface.inputs)
    replicated.inputs.push_back(replicatedSharding(
        llvm::cast<mlir::RankedTensorType>(port.type), report.mesh));
  replicated.outputs = {replicatedSharding(
      llvm::cast<mlir::RankedTensorType>(summary.interface.outputs[0].type),
      report.mesh)};
  auto conflict = replicated;
  conflict.inputs[1] = tensorLayoutChoices(
      llvm::cast<mlir::RankedTensorType>(summary.interface.inputs[1].type),
      report.mesh, {}, {std::nullopt, 1})[1];
  conflict.inputs[2] = tensorLayoutChoices(
      llvm::cast<mlir::RankedTensorType>(summary.interface.inputs[2].type),
      report.mesh, {}, {std::nullopt, 0})[1];
  conflict.outputs[0] = tensorLayoutChoices(
      llvm::cast<mlir::RankedTensorType>(summary.interface.outputs[0].type),
      report.mesh, {}, {std::nullopt, 1})[1];
  RegionEvaluator evaluator(report.mesh);
  auto original_r = evaluator.evaluate(*original, replicated),
       factored_r = evaluator.evaluate(*factored, replicated);
  auto original_conflict = evaluator.evaluate(*original, conflict),
       factored_conflict = evaluator.evaluate(*factored, conflict);
  ASSERT_TRUE(original_r.feasible && factored_r.feasible &&
              original_conflict.feasible && factored_conflict.feasible);
  EXPECT_LT(factored_r.cost.total(), original_r.cost.total());
  EXPECT_LT(original_conflict.cost.total(), factored_conflict.cost.total());
  EXPECT_GT(factored_conflict.cost.communication,
            original_conflict.cost.communication);
  bool checked_r = false, checked_conflict = false;
  for (const auto& plan : summary.plans) {
    if (plan.boundary == replicated) {
      EXPECT_EQ(plan.candidate_id, 1);
      checked_r = true;
    }
    if (plan.boundary == conflict) {
      EXPECT_EQ(plan.candidate_id, 0);
      checked_conflict = true;
    }
  }
  EXPECT_TRUE(checked_r && checked_conflict);
  for (const auto& pair : report.compositions[0].plans) {
    auto artifact = parse(pair.lowered_mlir);
    EXPECT_NEAR(CostModel().estimate(*artifact, report.mesh.mesh).total(),
                pair.cost.total(), 1e-9);
  }
}
TEST_F(PairCompositionTest, AdditionalInputsAndMultipleOutputsMaterialize) {
  auto module = parse(R"mlir(module {
    func.func @main(%x: tensor<8x4xf32>, %y: tensor<8x4xf32>, %z: tensor<8x4xf32>) -> (tensor<8x4xf32>,tensor<8x4xf32>) {
      %a=stablehlo.add %x,%y : tensor<8x4xf32>
      %b=stablehlo.multiply %a,%z : tensor<8x4xf32>
      %c=stablehlo.negate %b : tensor<8x4xf32>
      return %b,%c : tensor<8x4xf32>,tensor<8x4xf32>
    }
  })mlir");
  RegionOptimizerOptions options;
  options.regionizer = {2, {}};
  options.compose_regions = {{0, 1}};
  auto report = summarizeRegions(*module, options);
  ASSERT_EQ(report.regions.size(), 2);
  const auto& pair = report.compositions[0];
  ASSERT_EQ(pair.interface.external.inputs.size(), 3);
  ASSERT_EQ(pair.interface.external.outputs.size(), 2);
  EXPECT_EQ(pair.interface.b_inputs, (std::vector<int64_t>{-1, 2}));
  ASSERT_EQ(pair.plans.size(), 243);
  for (const auto& plan : pair.plans) {
    auto module = parse(plan.lowered_mlir);
    auto fn = module->lookupSymbol<mlir::func::FuncOp>("main");
    EXPECT_EQ(fn.getNumArguments(), 3);
    EXPECT_EQ(fn.getNumResults(), 2);
  }
}
TEST_F(PairCompositionTest, RejectsNonadjacentAndMissingRegionRequests) {
  auto module = chain();
  RegionOptimizerOptions options;
  options.compose_regions = {{1, 0}};
  EXPECT_THROW(summarizeRegions(*module, options), std::invalid_argument);
  options.compose_regions = {{0, 2}};
  EXPECT_THROW(summarizeRegions(*module, options), std::invalid_argument);
  auto fn = module->lookupSymbol<mlir::func::FuncOp>("main");
  auto regions = Regionizer().split(fn);
  ValueIndex values(*module);
  EXPECT_THROW(buildPairInterface(regions[1], regions[0], values),
               std::invalid_argument);
}
}  // namespace

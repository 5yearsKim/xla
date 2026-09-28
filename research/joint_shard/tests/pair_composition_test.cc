#include <map>

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"
#include "research/joint_shard/search/region_optimizer.h"
#include "research/joint_shard/sharding/region_evaluator.h"
#include "research/joint_shard/tests/support/mlir_test.h"
#include "shardy/dialect/sdy/ir/register.h"
#include "stablehlo/dialect/Register.h"

namespace joint_shard {

namespace {
class PairCompositionTest : public test::MlirTest {
 protected:
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
  a.finalizePlans();
  b.finalizePlans();
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
  a.finalizePlans();
  b.finalizePlans();
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
TEST_F(PairCompositionTest, InferredPairsAndReplacementRecipesMaterialize) {
  auto module = chain();
  auto before = print(*module);
  RegionOptimizerOptions options;
  options.compose_regions = {{0, 1}};
  auto report = summarizeRegions(*module, options);
  ASSERT_EQ(report.regions.size(), 2);
  ASSERT_EQ(report.compositions.size(), 1);
  const auto& pair = report.compositions[0];
  EXPECT_EQ(report.regions[0].input_states_evaluated, 9);
  EXPECT_EQ(report.regions[1].input_states_evaluated, 3);
  EXPECT_LE(pair.evaluations, 18 * 3);
  EXPECT_FALSE(pair.truncated);
  ASSERT_FALSE(pair.frontier.empty());
  bool replaced = false;
  for (size_t i = 0; i < pair.plans.size(); ++i) {
    const auto& plan = pair.plans[i];
    auto artifact = parse(materializePair(report.regions[0], report.regions[1],
                                          pair, i, report.mesh, CostModel{}));
    auto cost = CostModel().estimate(*artifact, report.mesh.mesh);
    EXPECT_NEAR(cost.compute, plan.cost.compute, 1e-9);
    EXPECT_NEAR(cost.communication, plan.cost.communication, 1e-9);
    if (plan.replacement) {
      replaced = true;
      EXPECT_TRUE(plan.lowered_mlir.empty());
      EXPECT_EQ(plan.a.input_adapters.size(), 0);
    } else
      EXPECT_FALSE(plan.lowered_mlir.empty());
  }
  EXPECT_TRUE(replaced);
  auto tampered = pair;
  tampered.plans[0].cost.compute += 1;
  EXPECT_THROW(materializePair(report.regions[0], report.regions[1], tampered,
                               0, report.mesh, CostModel{}),
               std::runtime_error);
  for (const auto& region : report.regions)
    for (const auto& witness : region.dominance) {
      EXPECT_TRUE(region.plans[witness.removed_id].lowered_mlir.empty());
      EXPECT_NE(std::find(region.frontier.begin(), region.frontier.end(),
                          witness.replacement_id),
                region.frontier.end());
    }
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
  RegionEvaluator evaluator(report.mesh);
  auto original_r = evaluator.evaluate(*original, replicated.inputs),
       factored_r = evaluator.evaluate(*factored, replicated.inputs);
  auto original_conflict = evaluator.evaluate(*original, conflict.inputs),
       factored_conflict = evaluator.evaluate(*factored, conflict.inputs);
  ASSERT_TRUE(original_r.feasible && factored_r.feasible &&
              original_conflict.feasible && factored_conflict.feasible);
  EXPECT_LT(factored_r.cost.total(), original_r.cost.total());
  EXPECT_LT(original_conflict.cost.total(), factored_conflict.cost.total());
  EXPECT_GT(factored_conflict.cost.communication,
            original_conflict.cost.communication);
  bool checked_r = false, checked_conflict = false;
  for (const auto& plan : summary.plans) {
    if (plan.boundary == factored_r.boundary) {
      EXPECT_EQ(plan.candidate_id, 1);
      checked_r = true;
    }
    if (plan.boundary == original_conflict.boundary) {
      EXPECT_EQ(plan.candidate_id, 0);
      checked_conflict = true;
    }
  }
  EXPECT_TRUE(checked_r && checked_conflict);
  const auto& pair = report.compositions[0];
  for (auto i : pair.frontier) {
    auto artifact = parse(pair.plans[i].lowered_mlir);
    EXPECT_NEAR(CostModel().estimate(*artifact, report.mesh.mesh).total(),
                pair.plans[i].cost.total(), 1e-9);
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
  ASSERT_FALSE(pair.plans.empty());
  EXPECT_LT(pair.plans.size(), 243);
  for (size_t i = 0; i < pair.plans.size(); ++i) {
    auto module = parse(materializePair(report.regions[0], report.regions[1],
                                        pair, i, report.mesh, CostModel{}));
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

}  // namespace joint_shard

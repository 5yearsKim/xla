#include "research/joint_shard/search/region_optimizer.h"

#include <limits>
#include <set>

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"
#include "shardy/dialect/sdy/ir/register.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace {
class RegionOptimizerTest : public ::testing::Test {
 protected:
  mlir::MLIRContext context;
  RegionOptimizerTest() {
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
  std::vector<Region> regions(mlir::ModuleOp module, size_t max = 128) {
    return Regionizer({max, {}}).split(
        module.lookupSymbol<mlir::func::FuncOp>("main"));
  }
  std::string print(mlir::ModuleOp module) {
    std::string text;
    llvm::raw_string_ostream out(text);
    module.print(out);
    return text;
  }
  mlir::OwningOpRef<mlir::ModuleOp> scalingDot() {
    return parse(R"mlir(module {
      func.func @main(%s: tensor<f32>, %a: tensor<8x16xf32>, %b: tensor<16x4xf32>) -> tensor<8x4xf32> {
        %scale = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f32>) -> tensor<8x16xf32>
        %scaled = stablehlo.multiply %scale, %a : tensor<8x16xf32>
        %dot = stablehlo.dot_general %scaled, %b, contracting_dims = [1] x [0]
            : (tensor<8x16xf32>, tensor<16x4xf32>) -> tensor<8x4xf32>
        return %dot : tensor<8x4xf32>
      }
    })mlir");
  }
};

TEST_F(RegionOptimizerTest, UnsupportedOperationsSplitWithoutDuplication) {
  auto module = parse(R"mlir(module {
    func.func @main(%x: tensor<8xf32>) -> tensor<8xf32> {
      %a = stablehlo.negate %x : tensor<8xf32>
      %b = stablehlo.sine %a : tensor<8xf32>
      %c = stablehlo.negate %b : tensor<8xf32>
      return %c : tensor<8xf32>
    }
  })mlir");
  auto split = regions(*module);
  ASSERT_EQ(split.size(), 2);
  EXPECT_EQ(split[0].operations.size(), 1);
  EXPECT_EQ(split[1].operations.size(), 1);
  EXPECT_NE(split[0].operations[0], split[1].operations[0]);
  EXPECT_EQ(split[0].outputs[0],
            split[1].inputs[0].getDefiningOp()->getOperand(0));
}

TEST_F(RegionOptimizerTest, SizeLimitAndCutUseDistinctCrossingValues) {
  auto module = parse(R"mlir(module {
    func.func @main(%x: tensor<8xf32>, %y: tensor<8xf32>) -> tensor<8xf32> {
      %a = stablehlo.add %x, %y : tensor<8xf32>
      %b = stablehlo.multiply %a, %x : tensor<8xf32>
      %c = stablehlo.subtract %a, %y : tensor<8xf32>
      %d = stablehlo.add %b, %c : tensor<8xf32>
      %e = stablehlo.multiply %d, %d : tensor<8xf32>
      %f = stablehlo.negate %e : tensor<8xf32>
      return %f : tensor<8xf32>
    }
  })mlir");
  auto whole = regions(*module);
  ASSERT_EQ(whole.size(), 1);
  EXPECT_EQ(countCrossingValues(whole[0].operations, 2), 2);
  EXPECT_EQ(countCrossingValues(whole[0].operations, 3), 2);
  EXPECT_EQ(countCrossingValues(whole[0].operations, 4), 1);
  auto split = regions(*module, 4);
  ASSERT_EQ(split.size(), 2);
  EXPECT_EQ(split[0].operations.size(), 4);  // Cheaper cut beats midpoint.
  EXPECT_EQ(split[1].operations.size(), 2);
  auto bounded = regions(*module, 2);
  size_t total = 0;
  for (const auto& region : bounded) {
    EXPECT_LE(region.operations.size(), 2);
    total += region.operations.size();
  }
  EXPECT_EQ(total, 6);
}

TEST_F(RegionOptimizerTest, ProtectsNonAdjacentTransposeDotAndReportsOversize) {
  auto module = parse(R"mlir(module {
    func.func @main(%a: tensor<8x16xf32>, %b: tensor<4x16xf32>) -> tensor<8x4xf32> {
      %t = stablehlo.transpose %b, dims = [1, 0] : (tensor<4x16xf32>) -> tensor<16x4xf32>
      %unused = stablehlo.negate %a : tensor<8x16xf32>
      %dot = stablehlo.dot_general %a, %t, contracting_dims = [1] x [0]
          : (tensor<8x16xf32>, tensor<16x4xf32>) -> tensor<8x4xf32>
      return %dot : tensor<8x4xf32>
    }
  })mlir");
  auto split = regions(*module, 1);
  ASSERT_EQ(split.size(), 1);
  EXPECT_TRUE(split[0].oversized);
  EXPECT_TRUE(cutBreaksProtectedPattern(split[0].operations, 1));
  EXPECT_TRUE(cutBreaksProtectedPattern(split[0].operations, 2));
}

TEST_F(RegionOptimizerTest, ProtectsReshapesAndBroadcastMultiplyDot) {
  auto module = scalingDot();
  auto split = regions(*module, 1);
  ASSERT_EQ(split.size(), 1);
  EXPECT_TRUE(split[0].oversized);
  auto reshapes = parse(R"mlir(module {
    func.func @main(%a: tensor<8x4xf32>) -> tensor<8x4xf32> {
      %r = stablehlo.reshape %a : (tensor<8x4xf32>) -> tensor<32xf32>
      %s = stablehlo.reshape %r : (tensor<32xf32>) -> tensor<8x4xf32>
      return %s : tensor<8x4xf32>
    }
  })mlir");
  EXPECT_EQ(regions(*reshapes, 1).size(), 1);
}

TEST_F(RegionOptimizerTest, InterfacesIncludeSharedValuesAndFunctionReturns) {
  auto module = parse(R"mlir(module {
    func.func @main(%x: tensor<8xf32>, %y: tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
      %a = stablehlo.add %x, %y : tensor<8xf32>
      %b = stablehlo.multiply %a, %a : tensor<8xf32>
      %c = stablehlo.negate %b : tensor<8xf32>
      return %a, %c : tensor<8xf32>, tensor<8xf32>
    }
  })mlir");
  auto whole = regions(*module)[0];
  auto first = describeRegion(llvm::ArrayRef(whole.operations).take_front(2));
  ASSERT_EQ(first.inputs.size(), 2);
  ASSERT_EQ(first.outputs.size(), 2);
  EXPECT_EQ(first.outputs[0], whole.operations[0]->getResult(0));
  EXPECT_EQ(first.outputs[1], whole.operations[1]->getResult(0));
  auto last = describeRegion(llvm::ArrayRef(whole.operations).drop_front(2));
  ASSERT_EQ(last.inputs.size(), 1);
  EXPECT_EQ(last.inputs[0], first.outputs[1]);
}

TEST_F(RegionOptimizerTest, SupportedRuleBlockersRemainEvaluatedSingletons) {
  auto module = parse(R"mlir(module {
    func.func @main(%x: tensor<8xf32>) -> tensor<8xf32> {
      %a = stablehlo.negate %x : tensor<8xf32>
      %b = stablehlo.tanh %a : tensor<8xf32>
      %c = stablehlo.negate %b : tensor<8xf32>
      return %c : tensor<8xf32>
    }
  })mlir");
  auto function = module->lookupSymbol<mlir::func::FuncOp>("main");
  EXPECT_EQ(Regionizer().split(function).size(), 3);
  EXPECT_EQ(regions(*module).size(), 1);
}

TEST_F(RegionOptimizerTest,
       RelaxedDefaultProducesAlternativesAndKeepsOriginal) {
  auto module = scalingDot();
  TensorRewriteOptions options;
  EXPECT_EQ(options.numerical_policy, NumericalPolicy::AllowReassociation);
  auto rules = compileTensorRules(options);
  auto saturated = saturateRegion(regions(*module)[0], rules, options);
  auto candidates = extractCandidates(saturated, options);
  ASSERT_GE(candidates.size(), 2);
  EXPECT_TRUE(candidates[0].is_original);
  EXPECT_EQ(candidates[0].name, "original");
  EXPECT_EQ(candidates[0].expression.nodes[candidates[0].output_roots[0]].op,
            OpKind::DotGeneral);
  bool scaledOutput = false;
  for (const auto& candidate : candidates)
    scaledOutput |= candidate.expression.nodes[candidate.output_roots[0]].op ==
                    OpKind::Multiply;
  EXPECT_TRUE(scaledOutput);
  EXPECT_EQ(extractCandidates(saturated, options, 1).size(), 1);
  EXPECT_THROW(extractCandidates(saturated, options, 0), std::invalid_argument);
  options.numerical_policy = NumericalPolicy::PreserveEvaluation;
  auto strict =
      saturateRegion(regions(*module)[0], compileTensorRules(options), options);
  EXPECT_EQ(extractCandidates(strict, options).size(), 1);
}

TEST_F(RegionOptimizerTest, BoundaryStatesAreExactTypedAndBounded) {
  auto module = scalingDot();
  auto region = regions(*module)[0];
  auto mesh = selectMesh(*module);
  auto all = enumerateBoundaryStates(region, mesh);
  EXPECT_EQ(all.states.size(), 27);  // scalar R; 3 choices for A, B, result.
  EXPECT_FALSE(all.truncated);
  for (const auto& state : all.states) {
    EXPECT_EQ(state.inputs[0].attr.getRank(), 0);
    for (const auto& input : state.inputs)
      EXPECT_TRUE(input.attr.isFullyClosed());
    for (const auto& output : state.outputs)
      EXPECT_TRUE(output.attr.isFullyClosed());
  }
  auto capped = enumerateBoundaryStates(region, mesh, {}, 5);
  EXPECT_EQ(capped.states.size(), 5);
  EXPECT_TRUE(capped.truncated);
  auto exactCap = enumerateBoundaryStates(region, mesh, {}, 27);
  EXPECT_FALSE(exactCap.truncated);
  auto vector =
      mlir::RankedTensorType::get({7}, mlir::Float32Type::get(&context));
  EXPECT_EQ(tensorLayoutChoices(vector, mesh, {}, {}).size(), 1);
}

TEST_F(RegionOptimizerTest, FixedCombinedLayoutIsPreservedOutsideCanonicalSet) {
  auto module = parse(R"mlir(module {
    sdy.mesh @mesh = <["data"=2, "model"=2]>
    func.func @main(%x: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>})
        -> (tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>}) {
      %a = stablehlo.negate %x : tensor<8x4xf32>
      return %a : tensor<8x4xf32>
    }
  })mlir");
  auto states =
      enumerateBoundaryStates(regions(*module)[0], selectMesh(*module));
  ASSERT_EQ(states.states.size(), 1);
  EXPECT_EQ(states.states[0].inputs[0], states.states[0].outputs[0]);
  EXPECT_EQ(states.states[0].inputs[0].attr.getDimSharding(0).getAxes().size(),
            1);
  EXPECT_EQ(states.states[0].inputs[0].attr.getDimSharding(1).getAxes().size(),
            1);
}

TEST_F(RegionOptimizerTest, PreservesDisjointShardingAndReplicatedSubAxes) {
  auto module = parse(R"mlir(module {
    sdy.mesh @mesh = <["model"=4]>
    func.func @main(%x: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model":(1)2}], replicated={"model":(2)2}>})
        -> tensor<8xf32> {
      %a = stablehlo.negate %x : tensor<8xf32>
      return %a : tensor<8xf32>
    }
  })mlir");
  auto mesh = selectMesh(*module);
  auto states = enumerateBoundaryStates(regions(*module)[0], mesh);
  ASSERT_FALSE(states.states.empty());
  for (const auto& state : states.states) {
    auto axes = state.inputs[0].attr.getDimSharding(0).getAxes();
    ASSERT_EQ(axes.size(), 1);
    EXPECT_EQ(axes[0].getSubAxisInfo().getSize(), 2);
  }
}

TEST_F(RegionOptimizerTest, OpenConstraintsPermitOnlyCompatibleRefinements) {
  auto module = parse(R"mlir(module {
    sdy.mesh @mesh = <["data"=2, "model"=2]>
    func.func @main(%x: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {?}]>}) -> tensor<8x4xf32> {
      %a = stablehlo.negate %x : tensor<8x4xf32>
      return %a : tensor<8x4xf32>
    }
  })mlir");
  auto states =
      enumerateBoundaryStates(regions(*module)[0], selectMesh(*module));
  ASSERT_EQ(states.states.size(), 6);
  for (const auto& state : states.states)
    EXPECT_TRUE(state.inputs[0].attr.getDimSharding(0).getAxes().empty());
}

TEST_F(RegionOptimizerTest, EvaluationReachesShardyAndPreservesOriginalModule) {
  auto module = scalingDot();
  auto before = print(*module);
  auto region = regions(*module)[0];
  auto mesh = selectMesh(*module);
  TensorRewriteOptions options;
  auto saturated = saturateRegion(region, compileTensorRules(options), options);
  auto prepared = prepareCandidateModule(region, saturated.original, mesh);
  auto choicesA = tensorLayoutChoices(
      llvm::cast<mlir::RankedTensorType>(region.inputs[1].getType()), mesh, {},
      {});
  auto choicesB = tensorLayoutChoices(
      llvm::cast<mlir::RankedTensorType>(region.inputs[2].getType()), mesh, {},
      {});
  auto choicesOut = tensorLayoutChoices(
      llvm::cast<mlir::RankedTensorType>(region.outputs[0].getType()), mesh, {},
      {});
  BoundaryState state{{replicatedSharding(llvm::cast<mlir::RankedTensorType>(
                                              region.inputs[0].getType()),
                                          mesh),
                       choicesA[1], choicesB[2]},
                      {choicesOut[1]}};
  auto result = RegionEvaluator(mesh).evaluate(*prepared, state);
  ASSERT_TRUE(result.feasible) << result.failure;
  ASSERT_EQ(result.snapshots.size(), 4);
  EXPECT_EQ(result.snapshots.back().communicationOps.at("sdy.all_gather"), 1);
  EXPECT_TRUE(result.cost.known());
  EXPECT_GT(result.cost.communication, 5);
  EXPECT_EQ(print(*module), before);
  auto preparedFunction = prepared->lookupSymbol<mlir::func::FuncOp>("main");
  EXPECT_FALSE(preparedFunction.getArgAttr(0, "sdy.sharding"));
  state.inputs.pop_back();
  EXPECT_FALSE(RegionEvaluator(mesh).evaluate(*prepared, state).feasible);
}

TEST_F(RegionOptimizerTest, MultiOutputCandidateKeepsSharedProducer) {
  auto module = parse(R"mlir(module {
    func.func @main(%x: tensor<8xf32>, %y: tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
      %a = stablehlo.add %x, %y : tensor<8xf32>
      %b = stablehlo.multiply %a, %a : tensor<8xf32>
      return %a, %b : tensor<8xf32>, tensor<8xf32>
    }
  })mlir");
  auto region = regions(*module)[0];
  TensorRewriteOptions options;
  auto saturated = saturateRegion(region, compileTensorRules(options), options);
  for (const auto& candidate : extractCandidates(saturated, options)) {
    auto prepared =
        prepareCandidateModule(region, candidate, selectMesh(*module));
    unsigned adds = 0;
    prepared->walk([&](mlir::stablehlo::AddOp) { ++adds; });
    EXPECT_EQ(adds, 1);
    EXPECT_EQ(candidate.output_roots.size(), 2);
  }
}

TEST_F(RegionOptimizerTest, OracleIsDirectionalAndCachesActualLowering) {
  auto module = scalingDot();
  auto mesh = selectMesh(*module);
  auto type =
      mlir::RankedTensorType::get({8, 4}, mlir::Float32Type::get(&context));
  auto choices = tensorLayoutChoices(type, mesh, {}, {});
  ReshardCostOracle oracle(mesh);
  auto slicing = oracle.estimate(choices[0], choices[1], type);
  auto gathering = oracle.estimate(choices[1], choices[0], type);
  ASSERT_TRUE(slicing.known());
  ASSERT_TRUE(gathering.known());
  EXPECT_EQ(slicing.total(), 0);
  EXPECT_GT(gathering.total(), 5);
  EXPECT_EQ(oracle.cacheSize(), 2);
  EXPECT_EQ(oracle.estimate(choices[1], choices[0], type).total(),
            gathering.total());
  EXPECT_EQ(oracle.cacheSize(), 2);
  EXPECT_EQ(oracle.estimate(choices[0], choices[0], type).total(), 0);
}

TEST_F(RegionOptimizerTest, ExactBoundarySelectionPreservesOriginalOnTies) {
  auto module = scalingDot();
  auto mesh = selectMesh(*module);
  auto type =
      mlir::RankedTensorType::get({8, 4}, mlir::Float32Type::get(&context));
  auto choices = tensorLayoutChoices(type, mesh, {}, {});
  BoundaryState first{{choices[0]}, {choices[0]}},
      second{{choices[1]}, {choices[1]}};
  RegionSummary summary;
  summary.plans = {{0, first, {12, 0, 0}, ""},
                   {1, first, {8, 0, 0}, ""},
                   {2, first, {10, 0, 0}, ""},
                   {0, second, {5, 0, 0}, ""},
                   {3, second, {5, 0, 0}, ""}};
  summary.keepBestPerBoundary();
  ASSERT_EQ(summary.plans.size(), 2);
  for (const auto& plan : summary.plans)
    EXPECT_EQ(plan.candidate_id, plan.boundary == first ? 1 : 0);
}

TEST_F(RegionOptimizerTest, UnknownEstimatesDoNotDisplaceKnownPlans) {
  auto module = scalingDot();
  auto state = enumerateBoundaryStates(regions(*module)[0], selectMesh(*module))
                   .states[0];
  RegionSummary summary;
  summary.plans = {{0, state, {0, 0, 1}, ""},
                   {1, state, {5, 0, 0}, ""},
                   {2, state, {0, 0, 1}, ""}};
  summary.keepBestPerBoundary();
  ASSERT_EQ(summary.plans.size(), 1);
  EXPECT_EQ(summary.plans[0].candidate_id, 1);
  EXPECT_TRUE(summary.plans[0].cost.known());
}

TEST_F(RegionOptimizerTest,
       DominanceUsesInputAndOutputDirectionsAndRetainedWitness) {
  auto module = scalingDot();
  auto mesh = selectMesh(*module);
  auto type =
      mlir::RankedTensorType::get({8, 4}, mlir::Float32Type::get(&context));
  auto choices = tensorLayoutChoices(type, mesh, {}, {});
  BoundaryState a{{choices[0]}, {choices[0]}}, b{{choices[1]}, {choices[1]}};
  RegionSummary summary;
  summary.interface.inputs = {{0, type}};
  summary.interface.outputs = {{1, type}};
  summary.plans = {{0, a, {10, 0, 0}, ""}, {1, b, {6, 0, 0}, ""}};
  std::vector<std::pair<TensorSharding, TensorSharding>> requests;
  summary.keepBestPerBoundary();
  pruneDominatedStates(summary, [&](auto from, auto to, auto tensor) {
    EXPECT_EQ(tensor, type);
    requests.emplace_back(from, to);
    return Cost{0, from == to ? 0.0 : 1.0, 0};
  });
  ASSERT_EQ(summary.frontier.size(), 1);
  EXPECT_EQ(summary.plans.at(summary.frontier[0]).boundary, b);
  EXPECT_EQ(summary.plans.size(), 2);
  ASSERT_EQ(summary.dominance.size(), 1);
  EXPECT_EQ(summary.dominance[0].replacement, b);
  EXPECT_EQ(summary.dominance[0].replacement_total, 8);
  auto witness_id = summary.dominance[0].replacement_id;
  EXPECT_EQ(summary.plans.at(witness_id).boundary, b);
  ASSERT_EQ(requests.size(), 2);
  EXPECT_EQ(requests[0], std::make_pair(choices[0], choices[1]));
  EXPECT_EQ(requests[1], std::make_pair(choices[1], choices[0]));
}

TEST_F(RegionOptimizerTest, TiesAndUnknownAdaptersDoNotDeleteEveryState) {
  auto module = scalingDot();
  auto mesh = selectMesh(*module);
  auto type =
      mlir::RankedTensorType::get({8, 4}, mlir::Float32Type::get(&context));
  auto choices = tensorLayoutChoices(type, mesh, {}, {});
  for (bool unknown : {false, true}) {
    RegionSummary summary;
    summary.interface.inputs = {{0, type}};
    summary.interface.outputs = {{1, type}};
    summary.plans = {{0, {{choices[0]}, {choices[0]}}, {1, 0, 0}, ""},
                     {1, {{choices[1]}, {choices[1]}}, {1, 0, 0}, ""}};
    summary.keepBestPerBoundary();
    pruneDominatedStates(summary, [&](auto, auto, auto) {
      return Cost{0, 0, unknown ? 1U : 0U};
    });
    EXPECT_EQ(summary.frontier.size(), unknown ? 2 : 1);
  }
}

TEST_F(RegionOptimizerTest, EndToEndBuildsFrontierWithoutMutatingSource) {
  auto module = scalingDot();
  auto before = print(*module);
  auto report = summarizeRegions(*module);
  ASSERT_EQ(report.regions.size(), 1);
  auto& summary = report.regions[0];
  EXPECT_GE(summary.candidates.size(), 2);
  EXPECT_EQ(summary.boundaries_evaluated, 27);
  EXPECT_EQ(summary.evaluations, 27 * summary.candidates.size());
  EXPECT_EQ(summary.best_boundary_plans, 27);
  EXPECT_EQ(summary.unknown_cost_plans, 0);
  EXPECT_FALSE(summary.plans.empty());
  EXPECT_LT(summary.frontier.size(), summary.best_boundary_plans);
  EXPECT_EQ(summary.plans.size(), summary.best_boundary_plans);
  for (const auto& plan : summary.plans) {
    EXPECT_TRUE(plan.cost.known());
    EXPECT_NE(plan.lowered_mlir.find("func.func @main"), std::string::npos);
  }
  EXPECT_EQ(print(*module), before);
  EXPECT_NE(report.str().find("After reshard dominance:"), std::string::npos);
}

TEST_F(RegionOptimizerTest, ValidatesRatesAndPartitionCount) {
  EXPECT_THROW(CostModel(CostModelOptions{0, 1, 0}), std::invalid_argument);
  EXPECT_THROW(CostModel(CostModelOptions{1, 1, -1}), std::invalid_argument);
  auto module = scalingDot();
  module->getOperation()->setAttr(
      "mhlo.num_partitions",
      mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 32), 8));
  EXPECT_THROW(summarizeRegions(*module), std::invalid_argument);
}
}  // namespace

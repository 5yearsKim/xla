#include "research/joint_shard/transforms/region_candidates.h"

#include <limits>
#include <set>

#include "research/joint_shard/tests/support/region_test.h"

namespace joint_shard {
namespace {
using RegionCandidatesTest = test::RegionTest;
TEST_F(RegionCandidatesTest,
       RelaxedDefaultProducesAlternativesAndKeepsOriginal) {
  auto module = scalingDot();
  TensorRewriteOptions options;
  EXPECT_EQ(options.numerical_policy, NumericalPolicy::AllowReassociation);
  auto rules = compileTensorRules(options);
  auto saturated = saturateRegion(regions(*module)[0], rules, options);
  auto candidates = extractCandidates(saturated, options);
  ASSERT_GE(candidates.size(), 2);
  EXPECT_EQ(candidates[0].id, 0);
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

TEST_F(RegionCandidatesTest, MultiOutputCandidateKeepsSharedProducer) {
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
    unsigned adds = 0;
    for (const auto& node : candidate.expression.nodes)
      if (node.op == OpKind::Add) ++adds;
    EXPECT_EQ(adds, 1);
    EXPECT_EQ(candidate.output_roots.size(), 2);
  }
}

}  // namespace

}  // namespace joint_shard

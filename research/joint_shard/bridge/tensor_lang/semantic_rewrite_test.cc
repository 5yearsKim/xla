#include <array>
#include <memory>
#include <random>
#include <stdexcept>

#include "mlir/IR/MLIRContext.h"
#include "eggc/runner.hpp"
#include "gtest/gtest.h"
#include "research/joint_shard/bridge/tensor_lang/tensor_patterns.h"

namespace joint_shard {
using patterns::attributes;

namespace {
class SemanticRewriteTest : public ::testing::Test {
 protected:
  mlir::MLIRContext context;
  mlir::RankedTensorType integer(std::vector<int64_t> shape = {2, 3}) {
    return mlir::RankedTensorType::get(shape,
                                       mlir::IntegerType::get(&context, 32));
  }
  eggc::Id input(TensorEGraph& graph, unsigned index,
                 mlir::RankedTensorType type) {
    return graph.add({OpKind::Input, InputAttrs{index, type}, {}});
  }
  SemanticRuleOptions reporting() {
    SemanticRuleOptions options;
    options.report =
        std::make_shared<std::map<std::string, SemanticRuleStats>>();
    return options;
  }
};
// Small independent numerical oracle for the integer matrix factorization
// gate. uint32_t arithmetic deliberately models modular i32 operations.
struct Matrix {
  unsigned rows, columns;
  std::vector<uint32_t> values;
  bool operator==(const Matrix&) const = default;
};
Matrix evaluateMatrix(const TensorRecExpr& expression,
                      const std::vector<Matrix>& inputs) {
  std::vector<Matrix> values;
  for (const auto& node : expression.nodes) {
    if (node.op == OpKind::Input) {
      values.push_back(inputs.at(std::get<InputAttrs>(node.attrs).index));
      continue;
    }
    const auto& a = values.at(node.operands.at(0));
    if (node.op == OpKind::BroadcastInDim) {
      const auto type = std::get<BroadcastAttrs>(node.attrs).result_type;
      if (a.values.size() != 1 || type.getRank() != 2 ||
          !std::get<BroadcastAttrs>(node.attrs).dimensions.empty())
        throw std::logic_error(
            "oracle requires a scalar broadcast to a matrix");
      values.push_back(
          {static_cast<unsigned>(type.getDimSize(0)),
           static_cast<unsigned>(type.getDimSize(1)),
           std::vector<uint32_t>(type.getNumElements(), a.values[0])});
      continue;
    }
    if (node.op == OpKind::Transpose) {
      if (std::get<TransposeAttrs>(node.attrs).permutation !=
          std::vector<int64_t>{1, 0})
        throw std::logic_error("oracle requires a matrix transpose");
      Matrix result{a.columns, a.rows, std::vector<uint32_t>(a.values.size())};
      for (unsigned i = 0; i < a.rows; ++i)
        for (unsigned j = 0; j < a.columns; ++j)
          result.values[j * a.rows + i] = a.values[i * a.columns + j];
      values.push_back(std::move(result));
      continue;
    }
    const auto& b = values.at(node.operands.at(1));
    if (node.op == OpKind::Add || node.op == OpKind::Multiply) {
      if (a.rows != b.rows || a.columns != b.columns)
        throw std::logic_error("oracle add shape mismatch");
      auto result = a;
      for (size_t i = 0; i < a.values.size(); ++i)
        result.values[i] = node.op == OpKind::Add ? a.values[i] + b.values[i]
                                                  : a.values[i] * b.values[i];
      values.push_back(std::move(result));
    } else if (node.op == OpKind::DotGeneral) {
      const auto& attrs = std::get<DotGeneralAttrs>(node.attrs);
      if (attrs.dimensions.lhs_contracting != std::vector<int64_t>{1} ||
          attrs.dimensions.rhs_contracting != std::vector<int64_t>{0} ||
          !attrs.dimensions.lhs_batching.empty() ||
          !attrs.dimensions.rhs_batching.empty() || a.columns != b.rows)
        throw std::logic_error("oracle dot signature mismatch");
      Matrix result{a.rows, b.columns,
                    std::vector<uint32_t>(a.rows * b.columns)};
      for (unsigned i = 0; i < a.rows; ++i)
        for (unsigned j = 0; j < b.columns; ++j)
          for (unsigned k = 0; k < a.columns; ++k)
            result.values[i * b.columns + j] +=
                a.values[i * a.columns + k] * b.values[k * b.columns + j];
      values.push_back(std::move(result));
    } else
      throw std::logic_error("unsupported numerical oracle node");
  }
  return values.back();
}
TEST_F(SemanticRewriteTest, ConjunctionUsesSharedProperties) {
  const auto pattern_F = patterns::operator_var("F", 2);
  const auto pattern_x = patterns::tensor_var("x");
  const auto pattern_y = patterns::tensor_var("y");

  TensorEGraph graph;
  auto x = input(graph, 0, integer()), y = input(graph, 1, integer());
  auto root = graph.add({OpKind::Add, NoAttrs{}, {x, y}});
  auto rules =
      compileRules({patterns::rule("swap", pattern_F(pattern_x, pattern_y),
                                   pattern_F(pattern_y, pattern_x))
                        .when({patterns::elementwise(pattern_F),
                               patterns::commutative(pattern_F)})});
  eggc::run(graph, rules);
  auto reverse = graph.add({OpKind::Add, NoAttrs{}, {y, x}});
  EXPECT_EQ(graph.find(root), graph.find(reverse));
}
TEST_F(SemanticRewriteTest, StrictFloatingReorderingIsRejectedWithReason) {
  const auto pattern_F = patterns::operator_var("F", 2);
  const auto pattern_x = patterns::tensor_var("x");
  const auto pattern_y = patterns::tensor_var("y");

  TensorEGraph graph;
  auto type =
      mlir::RankedTensorType::get({2, 3}, mlir::Float32Type::get(&context));
  auto x = input(graph, 0, type), y = input(graph, 1, type);
  graph.add({OpKind::Add, NoAttrs{}, {x, y}});
  auto options = reporting();
  auto count = graph.node_count();
  eggc::run(graph, compileRules(
                       {patterns::rule("swap", pattern_F(pattern_x, pattern_y),
                                       pattern_F(pattern_y, pattern_x))
                            .when({patterns::commutative(pattern_F)})},
                       NumericalPolicy::PreserveEvaluation, options));
  EXPECT_EQ(graph.node_count(), count);
  EXPECT_GT(options.report->at("swap").rejections.size(), 0);
  EXPECT_EQ(options.report->at("swap").applied, 0);
}
TEST_F(SemanticRewriteTest,
       IndividualNumericalPermissionsDoNotEnableOtherLaws) {
  auto type =
      mlir::RankedTensorType::get({2, 3}, mlir::Float32Type::get(&context));
  std::array<TensorFacts, 2> operands{{{type, {}}, {type, {}}}};
  TensorNode add{OpKind::Add, NoAttrs{}, {0, 1}};
  NumericalPermissions permissions;
  permissions.reorder_floating_point = true;
  PropertyContext facts{
      operands, {type, {}}, NumericalPolicy::PreserveEvaluation, permissions};
  EXPECT_FALSE(queryProperty(add, Commutative{}, facts).allowed);
  permissions.assume_finite = true;
  permissions.ignore_signed_zero = true;
  facts.permissions = permissions;
  EXPECT_TRUE(queryProperty(add, Commutative{}, facts).allowed);
  EXPECT_FALSE(queryProperty(add, Associative{}, facts).allowed);
}
TEST_F(SemanticRewriteTest, InvalidRhsMakesNoPartialInsertion) {
  const auto pattern_F = patterns::operator_var("F", 1);
  const auto pattern_x = patterns::tensor_var("x");

  TensorEGraph graph;
  auto x = input(graph, 0, integer({2, 3}));
  auto f = graph.add({OpKind::Transpose, TransposeAttrs{{1, 0}}, {x}});
  graph.add({OpKind::Transpose, TransposeAttrs{{1, 0}}, {f}});
  auto options = reporting();
  auto count = graph.node_count();
  // negate(?x) would insert a node before the add discovered its shape mismatch
  // in the old recursive instantiator. Preparation rejects the entire RHS.
  auto rules = compileRules(
      {patterns::rule("mismatch", pattern_F(pattern_F(pattern_x)),
                      tensorlang::ops::add(tensorlang::ops::negate(pattern_x),
                                           pattern_F(pattern_x)))
           .when({patterns::involution(pattern_F)})},
      NumericalPolicy::PreserveEvaluation, options);
  eggc::run(graph, rules, 1);
  EXPECT_EQ(graph.node_count(), count);
  EXPECT_GT(options.report->at("mismatch").rejections.size(), 0);
}
TEST_F(SemanticRewriteTest, BoundedSearchReportsSearchLimit) {
  TensorEGraph graph;
  auto x = input(graph, 0, integer()), y = input(graph, 1, integer());
  graph.add({OpKind::Add, NoAttrs{}, {x, y}});
  auto options = reporting();
  options.visit_limit = 1;
  auto result = eggc::run(
      graph, buildSemanticRules(NumericalPolicy::PreserveEvaluation, options));
  EXPECT_EQ(result.reason, eggc::StopReason::SearchLimit);
  EXPECT_GT(options.report->at("commute-binary-operator").budget_stops, 0);
}
TEST_F(SemanticRewriteTest, DotLinearityFactorsTwoDots) {
  const auto pattern_F = patterns::operator_var("F", 2);
  const auto pattern_a = patterns::tensor_var("a");
  const auto pattern_b = patterns::tensor_var("b");
  const auto pattern_c = patterns::tensor_var("c");

  TensorEGraph graph;
  auto a = input(graph, 0, integer({2, 3})),
       b = input(graph, 1, integer({3, 4}));
  auto c = input(graph, 2, integer({3, 4}));
  DotGeneralAttrs attrs{
      DotDimensions{{1}, {0}, {}, {}}, {}, {}, integer({2, 4}), {}};
  auto ab = graph.add({OpKind::DotGeneral, attrs, {a, b}});
  auto ac = graph.add({OpKind::DotGeneral, attrs, {a, c}});
  auto root = graph.add({OpKind::Add, NoAttrs{}, {ab, ac}});
  graph.rebuild();
  auto original = TensorExtractor(graph).find_best(root).second;
  auto rules = compileRules(
      {patterns::rule(
           "factor",
           tensorlang::ops::add(pattern_F(pattern_a, pattern_b),
                                pattern_F(pattern_a, pattern_c)),
           pattern_F(pattern_a, tensorlang::ops::add(pattern_b, pattern_c)))
           .when({patterns::linear_in(pattern_F, 1)})});
  eggc::run(graph, rules);
  auto bc = graph.add({OpKind::Add, NoAttrs{}, {b, c}});
  auto factored = graph.add({OpKind::DotGeneral, attrs, {a, bc}});
  EXPECT_EQ(graph.find(root), graph.find(factored));
  graph.rebuild();
  auto cost = [](const TensorNode& node,
                 const std::vector<size_t>& children) -> std::optional<size_t> {
    size_t value = node.op == OpKind::DotGeneral ? 100 : 1;
    for (auto child : children) value += child;
    return value;
  };
  auto optimized = TensorExtractor(graph, cost).find_best(root).second;
  unsigned dots = 0;
  for (const auto& node : optimized.nodes)
    if (node.op == OpKind::DotGeneral) ++dots;
  EXPECT_EQ(dots, 1);
  std::mt19937 random(42);
  for (unsigned trial = 0; trial < 16; ++trial) {
    std::vector<Matrix> inputs{{2, 3, std::vector<uint32_t>(6)},
                               {3, 4, std::vector<uint32_t>(12)},
                               {3, 4, std::vector<uint32_t>(12)}};
    for (auto& matrix : inputs)
      for (auto& value : matrix.values) value = random();
    EXPECT_EQ(evaluateMatrix(original, inputs),
              evaluateMatrix(optimized, inputs));
  }
}
TEST_F(SemanticRewriteTest, ArbitraryTensorScaleDoesNotProveHomogeneity) {
  const auto pattern_F = patterns::operator_var("F", 1);
  const auto pattern_scale = patterns::tensor_var("scale");
  const auto pattern_x = patterns::tensor_var("x");

  TensorEGraph graph;
  auto x = input(graph, 0, integer({2, 2})),
       scale = input(graph, 1, integer({2, 2}));
  auto mul = graph.add({OpKind::Multiply, NoAttrs{}, {scale, x}});
  graph.add({OpKind::Transpose, TransposeAttrs{{1, 0}}, {mul}});
  auto options = reporting();
  auto count = graph.node_count();
  auto rules = compileRules(
      {patterns::rule(
           "scale",
           pattern_F(tensorlang::ops::multiply(pattern_scale, pattern_x)),
           tensorlang::ops::multiply(pattern_scale, pattern_F(pattern_x)))
           .when({patterns::homogeneous_in(pattern_F, 0),
                  patterns::uniform(pattern_scale)})},
      NumericalPolicy::PreserveEvaluation, options);
  eggc::run(graph, rules);
  EXPECT_EQ(graph.node_count(), count);
  EXPECT_GT(options.report->at("scale").rejections.size(), 0);
}
// Prefer moving scale out so the numerical oracle examines the constructed
// RHS rather than an equivalent original expression selected on a cost tie.
auto preferUnscaledOperators(const TensorEGraph& graph) {
  return
      [&graph](const TensorNode& node,
               const std::vector<size_t>& children) -> std::optional<size_t> {
        size_t cost = 1;
        for (auto child : children) cost += child;
        if (node.op == OpKind::Transpose || node.op == OpKind::DotGeneral)
          for (auto child : node.operands)
            for (const auto& operand : graph.nodes(child))
              if (operand.op == OpKind::Multiply) cost += 10000;
        return cost;
      };
}
TEST_F(SemanticRewriteTest, ScaleRebuildsBroadcastForRectangularTranspose) {
  const auto pattern_F = patterns::operator_var("F", 1);
  const auto pattern_s = patterns::tensor_var("s");
  const auto pattern_x = patterns::tensor_var("x");

  for (bool reverse : {false, true}) {
    TensorEGraph graph;
    auto scalar = input(graph, 0, integer({}));
    auto x = input(graph, 1, integer({2, 3}));
    auto broadcast = graph.add({OpKind::BroadcastInDim,
                                BroadcastAttrs{{}, integer({2, 3})},
                                {scalar}});
    auto multiply = graph.add({OpKind::Multiply, NoAttrs{},
                               reverse ? std::vector<eggc::Id>{x, broadcast}
                                       : std::vector<eggc::Id>{broadcast, x}});
    auto root =
        graph.add({OpKind::Transpose, TransposeAttrs{{1, 0}}, {multiply}});
    graph.rebuild();
    auto original = TensorExtractor(graph).find_best(root).second;
    auto rules = compileRules(
        {patterns::rule("move",
                        pattern_F(patterns::scale(pattern_s, pattern_x)),
                        patterns::scale(pattern_s, pattern_F(pattern_x)))
             .when({patterns::homogeneous_in(pattern_F, 0),
                    patterns::scalar(pattern_s)})});
    eggc::run(graph, rules);
    auto transposed =
        graph.add({OpKind::Transpose, TransposeAttrs{{1, 0}}, {x}});
    auto output_scale = graph.add({OpKind::BroadcastInDim,
                                   BroadcastAttrs{{}, integer({3, 2})},
                                   {scalar}});
    auto expected =
        graph.add({OpKind::Multiply, NoAttrs{}, {output_scale, transposed}});
    EXPECT_EQ(graph.find(root), graph.find(expected));
    graph.rebuild();
    auto optimized = TensorExtractor(graph, preferUnscaledOperators(graph))
                         .find_best(root)
                         .second;
    EXPECT_EQ(optimized.nodes.back().op, OpKind::Multiply);
    std::vector<Matrix> inputs{{1, 1, {7}}, {2, 3, {1, 2, 3, 4, 5, 6}}};
    EXPECT_EQ(evaluateMatrix(original, inputs),
              evaluateMatrix(optimized, inputs));
  }
}
TEST_F(SemanticRewriteTest,
       ScaleRebuildsBroadcastForBothRectangularDotOperands) {
  for (unsigned slot : {0u, 1u}) {
    TensorEGraph graph;
    auto scalar = input(graph, 0, integer({}));
    auto a = input(graph, 1, integer({2, 3}));
    auto b = input(graph, 2, integer({3, 4}));
    auto scale_type = slot == 0 ? integer({2, 3}) : integer({3, 4});
    auto broadcast = graph.add(
        {OpKind::BroadcastInDim, BroadcastAttrs{{}, scale_type}, {scalar}});
    auto scaled = graph.add(
        {OpKind::Multiply, NoAttrs{}, {broadcast, slot == 0 ? a : b}});
    DotGeneralAttrs attrs{
        DotDimensions{{1}, {0}, {}, {}}, {}, {}, integer({2, 4}), {}};
    auto root = graph.add({OpKind::DotGeneral, attrs,
                           slot == 0 ? std::vector<eggc::Id>{scaled, b}
                                     : std::vector<eggc::Id>{a, scaled}});
    graph.rebuild();
    auto original = TensorExtractor(graph).find_best(root).second;
    auto rules = buildSemanticRules();
    eggc::run(graph, rules);
    auto dot = graph.add({OpKind::DotGeneral, attrs, {a, b}});
    auto output_scale = graph.add({OpKind::BroadcastInDim,
                                   BroadcastAttrs{{}, integer({2, 4})},
                                   {scalar}});
    auto expected =
        graph.add({OpKind::Multiply, NoAttrs{}, {output_scale, dot}});
    EXPECT_EQ(graph.find(root), graph.find(expected));
    graph.rebuild();
    auto optimized = TensorExtractor(graph, preferUnscaledOperators(graph))
                         .find_best(root)
                         .second;
    EXPECT_EQ(optimized.nodes.back().op, OpKind::Multiply);
    std::mt19937 random(42);
    for (unsigned trial = 0; trial < 16; ++trial) {
      std::vector<Matrix> inputs{{1, 1, {0}},
                                 {2, 3, std::vector<uint32_t>(6)},
                                 {3, 4, std::vector<uint32_t>(12)}};
      for (auto& matrix : inputs)
        for (auto& value : matrix.values) value = random();
      EXPECT_EQ(evaluateMatrix(original, inputs),
                evaluateMatrix(optimized, inputs));
    }
  }
}
TEST_F(SemanticRewriteTest,
       ScalarPredicateUsesAnyVariableNameAndRejectsTensors) {
  const auto pattern_F = patterns::operator_var("F", 1);
  const auto pattern_coefficient = patterns::tensor_var("coefficient");

  auto rules = compileRules(
      {patterns::rule("scalar", pattern_F(pattern_F(pattern_coefficient)),
                      pattern_coefficient)
           .when({patterns::involution(pattern_F),
                  patterns::scalar(pattern_coefficient)})});
  TensorEGraph graph;
  auto x = input(graph, 0, integer({2, 3}));
  auto negated = graph.add({OpKind::Negate, NoAttrs{}, {x}});
  auto root = graph.add({OpKind::Negate, NoAttrs{}, {negated}});
  auto scalar = input(graph, 1, integer({}));
  auto scalar_negated = graph.add({OpKind::Negate, NoAttrs{}, {scalar}});
  auto scalar_root = graph.add({OpKind::Negate, NoAttrs{}, {scalar_negated}});
  eggc::run(graph, rules);
  EXPECT_NE(graph.find(root), graph.find(x));
  EXPECT_EQ(graph.find(scalar_root), graph.find(scalar));
}
TEST_F(SemanticRewriteTest,
       ScaleRequiresScalarProvenanceAndRespectsSearchBudget) {
  const auto pattern_F = patterns::operator_var("F", 1);
  const auto pattern_s = patterns::tensor_var("s");
  const auto pattern_x = patterns::tensor_var("x");

  TensorEGraph graph;
  auto x = input(graph, 0, integer({2, 3}));
  auto tensor_scale = input(graph, 1, integer({2, 3}));
  auto multiply = graph.add({OpKind::Multiply, NoAttrs{}, {tensor_scale, x}});
  graph.add({OpKind::Transpose, TransposeAttrs{{1, 0}}, {multiply}});
  auto options = reporting();
  auto rules = compileRules(
      {patterns::rule("move", pattern_F(patterns::scale(pattern_s, pattern_x)),
                      patterns::scale(pattern_s, pattern_F(pattern_x)))
           .when({patterns::homogeneous_in(pattern_F, 0),
                  patterns::scalar(pattern_s)})},
      NumericalPolicy::PreserveEvaluation, options);
  auto count = graph.node_count();
  eggc::run(graph, rules);
  EXPECT_EQ(graph.node_count(), count);
  EXPECT_EQ(options.report->at("move").structural_matches, 0);

  // Budget expires inside scalar provenance lookup before a match is emitted.
  TensorEGraph bounded;
  auto scalar = input(bounded, 0, integer({}));
  auto value = input(bounded, 1, integer({2, 3}));
  auto broadcast = bounded.add(
      {OpKind::BroadcastInDim, BroadcastAttrs{{}, integer({2, 3})}, {scalar}});
  bounded.add({OpKind::Multiply, NoAttrs{}, {broadcast, value}});
  options.visit_limit = 2;
  auto limited = compileRules(
      {patterns::rule("bounded", patterns::scale(pattern_s, pattern_x),
                      patterns::scale(pattern_s, pattern_x))
           .when({patterns::scalar(pattern_s)})},
      NumericalPolicy::PreserveEvaluation, options);
  auto result = eggc::run(bounded, limited);
  EXPECT_EQ(result.reason, eggc::StopReason::SearchLimit);
  EXPECT_GT(options.report->at("bounded").budget_stops, 0);
}
TEST_F(SemanticRewriteTest, StrictFloatScaleDoesNotReorderMultiplyOperands) {
  const auto pattern_F = patterns::operator_var("F", 1);
  const auto pattern_s = patterns::tensor_var("s");
  const auto pattern_x = patterns::tensor_var("x");

  auto scalar_type =
      mlir::RankedTensorType::get({}, mlir::Float32Type::get(&context));
  auto matrix_type =
      mlir::RankedTensorType::get({2, 3}, mlir::Float32Type::get(&context));
  for (bool relaxed : {false, true}) {
    TensorEGraph graph;
    auto scalar = input(graph, 0, scalar_type);
    auto x = input(graph, 1, matrix_type);
    auto broadcast = graph.add(
        {OpKind::BroadcastInDim, BroadcastAttrs{{}, matrix_type}, {scalar}});
    auto multiply = graph.add({OpKind::Multiply, NoAttrs{}, {x, broadcast}});
    auto root =
        graph.add({OpKind::Transpose, TransposeAttrs{{1, 0}}, {multiply}});
    auto options = reporting();
    auto rules = compileRules(
        {patterns::rule("move",
                        pattern_F(patterns::scale(pattern_s, pattern_x)),
                        patterns::scale(pattern_s, pattern_F(pattern_x)))
             .when({patterns::homogeneous_in(pattern_F, 0),
                    patterns::scalar(pattern_s)})},
        relaxed ? NumericalPolicy::AllowReassociation
                : NumericalPolicy::PreserveEvaluation,
        options);
    auto count = graph.node_count();
    eggc::run(graph, rules);
    if (relaxed) {
      EXPECT_GT(options.report->at("move").applied, 0);
      EXPECT_GT(graph.node_count(), count);
      EXPECT_EQ(tensorFacts(graph, root).type.getShape()[0], 3);
    } else {
      EXPECT_EQ(options.report->at("move").structural_matches, 0);
      EXPECT_EQ(graph.node_count(), count);
    }
  }
}
TEST_F(SemanticRewriteTest, ScaleFollowsNestedScalarBroadcasts) {
  const auto pattern_F = patterns::operator_var("F", 1);
  const auto pattern_s = patterns::tensor_var("s");
  const auto pattern_x = patterns::tensor_var("x");

  TensorEGraph graph;
  auto scalar = input(graph, 0, integer({}));
  auto x = input(graph, 1, integer({2, 3}));
  auto first = graph.add(
      {OpKind::BroadcastInDim, BroadcastAttrs{{}, integer({3})}, {scalar}});
  auto second = graph.add(
      {OpKind::BroadcastInDim, BroadcastAttrs{{1}, integer({2, 3})}, {first}});
  auto multiply = graph.add({OpKind::Multiply, NoAttrs{}, {second, x}});
  auto root =
      graph.add({OpKind::Transpose, TransposeAttrs{{1, 0}}, {multiply}});
  auto rules = compileRules(
      {patterns::rule("move", pattern_F(patterns::scale(pattern_s, pattern_x)),
                      patterns::scale(pattern_s, pattern_F(pattern_x)))
           .when({patterns::homogeneous_in(pattern_F, 0),
                  patterns::scalar(pattern_s)})});
  eggc::run(graph, rules);
  auto transpose = graph.add({OpKind::Transpose, TransposeAttrs{{1, 0}}, {x}});
  auto broadcast = graph.add(
      {OpKind::BroadcastInDim, BroadcastAttrs{{}, integer({3, 2})}, {scalar}});
  auto expected =
      graph.add({OpKind::Multiply, NoAttrs{}, {broadcast, transpose}});
  EXPECT_EQ(graph.find(root), graph.find(expected));
}
TEST_F(SemanticRewriteTest, UniformPredicateAcceptsBroadcastOfScalar) {
  const auto pattern_F = patterns::operator_var("F", 1);
  const auto pattern_value = patterns::tensor_var("value");

  TensorEGraph graph;
  auto scalar = input(graph, 0, integer({}));
  auto uniform = graph.add(
      {OpKind::BroadcastInDim, BroadcastAttrs{{}, integer({2, 3})}, {scalar}});
  auto negate = graph.add({OpKind::Negate, NoAttrs{}, {uniform}});
  auto root = graph.add({OpKind::Negate, NoAttrs{}, {negate}});
  auto rules = compileRules(
      {patterns::rule("uniform", pattern_F(pattern_F(pattern_value)),
                      pattern_value)
           .when({patterns::involution(pattern_F),
                  patterns::uniform(pattern_value)})});
  eggc::run(graph, rules);
  EXPECT_EQ(graph.find(root), graph.find(uniform));
}
TEST_F(SemanticRewriteTest, ScaleOnRankZeroNeedsNoBroadcast) {
  const auto pattern_F = patterns::operator_var("F", 1);
  const auto pattern_s = patterns::tensor_var("s");
  const auto pattern_x = patterns::tensor_var("x");

  TensorEGraph graph;
  auto s = input(graph, 0, integer({}));
  auto x = input(graph, 1, integer({}));
  auto product = graph.add({OpKind::Multiply, NoAttrs{}, {s, x}});
  auto root = graph.add({OpKind::Negate, NoAttrs{}, {product}});
  auto rules = compileRules(
      {patterns::rule("move", pattern_F(patterns::scale(pattern_s, pattern_x)),
                      patterns::scale(pattern_s, pattern_F(pattern_x)))
           .when({patterns::homogeneous_in(pattern_F, 0),
                  patterns::scalar(pattern_s)})});
  eggc::run(graph, rules);
  auto negated = graph.add({OpKind::Negate, NoAttrs{}, {x}});
  auto expected = graph.add({OpKind::Multiply, NoAttrs{}, {s, negated}});
  EXPECT_EQ(graph.find(root), graph.find(expected));
  EXPECT_TRUE(graph.classes_for_op(OpKind::BroadcastInDim).empty());
}
TEST_F(SemanticRewriteTest, NonScalarRhsScaleLeavesGraphUnchanged) {
  const auto pattern_F = patterns::operator_var("F", 2);
  const auto pattern_s = patterns::tensor_var("s");
  const auto pattern_x = patterns::tensor_var("x");

  TensorEGraph graph;
  auto a = input(graph, 0, integer({2, 3}));
  auto b = input(graph, 1, integer({2, 3}));
  graph.add({OpKind::Add, NoAttrs{}, {a, b}});
  auto options = reporting();
  auto rules = compileRules(
      {patterns::rule(
           "bad", pattern_F(pattern_s, pattern_x),
           patterns::scale(pattern_s, tensorlang::ops::negate(pattern_x)))
           .when({patterns::elementwise(pattern_F)})},
      NumericalPolicy::PreserveEvaluation, options);
  auto count = graph.node_count();
  eggc::run(graph, rules);
  EXPECT_EQ(graph.node_count(), count);
  EXPECT_GT(options.report->at("bad").rejections.size(), 0);
}
TEST_F(SemanticRewriteTest, ComposesNonInvolutivePermutation) {
  TensorEGraph graph;
  auto x = input(graph, 0, integer({2, 3, 4}));
  auto p = graph.add({OpKind::Transpose, TransposeAttrs{{1, 2, 0}}, {x}});
  auto q = graph.add({OpKind::Transpose, TransposeAttrs{{2, 0, 1}}, {p}});
  eggc::run(graph, buildAttributeRewrites());
  EXPECT_EQ(graph.find(x), graph.find(q));
}
TEST_F(SemanticRewriteTest, StandaloneInferenceDistinguishesUnknownAndInvalid) {
  std::array<TensorFacts, 1> unknown{};
  EXPECT_EQ(inferTensorNode({OpKind::Negate, NoAttrs{}, {0}}, unknown).status,
            InferenceStatus::Unknown);
  std::array<TensorFacts, 1> known{{{integer({2, 3}), {}}}};
  EXPECT_EQ(
      inferTensorNode({OpKind::Transpose, TransposeAttrs{{0, 0}}, {0}}, known)
          .status,
      InferenceStatus::Invalid);
}
// Concrete operators reject invalid calls at compile time; operator variables
// still validate their manually declared arity at runtime.
static_assert(std::is_invocable_v<decltype(tensorlang::ops::add),
                                  patterns::TensorVar, patterns::TensorVar>);
static_assert(
    !std::is_invocable_v<decltype(tensorlang::ops::add), patterns::TensorVar>);
static_assert(
    !std::is_invocable_v<decltype(tensorlang::ops::add), patterns::TensorVar,
                         patterns::TensorVar, patterns::TensorVar>);
static_assert(!std::is_invocable_v<decltype(tensorlang::ops::transpose),
                                   patterns::TensorVar,
                                   patterns::attributes<ReduceAttrs>>);
static_assert(!std::is_constructible_v<patterns::Attribute<Axes>,
                                       patterns::AttributeVar<ReduceKind>>);

TEST_F(SemanticRewriteTest, SchemaDerivedMaximumNeedsNoHandwrittenPattern) {
  using namespace patterns;
  const auto x = tensor_var("x");
  TensorEGraph graph;
  auto value = input(graph, 0, integer({2, 3}));
  auto root = graph.add({OpKind::Maximum, NoAttrs{}, {value, value}});
  eggc::run(graph, compileRules({rule("idempotent-maximum",
                                      tensorlang::ops::maximum(x, x), x)}));
  EXPECT_EQ(graph.find(root), graph.find(value));
}

TEST_F(SemanticRewriteTest, CppDefinitionsValidateBindingsAndArity) {
  using namespace patterns;
  using namespace tensorlang::ops;
  const auto x = tensor_var("x"), y = tensor_var("y");
  const auto F = operator_var("F", 1), G = operator_var("G", 1);
  EXPECT_THROW(F(x, y), std::invalid_argument);
  EXPECT_THROW(linear_in(F, 1), std::invalid_argument);
  EXPECT_THROW(compileRules({rule("unbound", F(x), y)}), std::invalid_argument);
  EXPECT_THROW(compileRules({rule("unbound-op", F(x), G(x))}),
               std::invalid_argument);
  EXPECT_THROW(compileRules({rule("bad-guard", F(x), x).when(scalar(y))}),
               std::invalid_argument);
  EXPECT_THROW(
      compileRules({rule("bad-arity", F(x), operator_var("F", 2)(x, x))}),
      std::invalid_argument);
  EXPECT_THROW(compileRules({rule("missing-rhs", F(x))}),
               std::invalid_argument);
  EXPECT_THROW(compileRules({rule("bare-lhs", x, x)}), std::invalid_argument);
  const auto good = rule("duplicate", F(F(x)), x).when(involution(F));
  EXPECT_THROW(compileRules({good, good}), std::invalid_argument);
  auto options = reporting();
  options.enable_linearity = false;
  // Validation still checks disabled definitions.
  EXPECT_THROW(
      compileRules({rule("disabled-invalid", F(x), y).when(linear_in(F, 0))},
                   NumericalPolicy::PreserveEvaluation, options),
      std::invalid_argument);
  try {
    compileRules({rule("diagnostic-name", F(x), y)});
    FAIL();
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what()).find("diagnostic-name"),
              std::string::npos);
  }
}
TEST_F(SemanticRewriteTest, CopiedDefinitionsHaveIndependentGuards) {
  using namespace patterns;
  using namespace tensorlang::ops;
  auto x = tensor_var("x");
  auto F = operator_var("F", 1);
  auto original = rule("cancel", F(F(x)), x).when(involution(F));
  auto restricted = original;
  restricted.when(scalar(x));
  for (bool restrict : {false, true}) {
    TensorEGraph graph;
    auto value = input(graph, 0, integer());
    auto inner = graph.add({OpKind::Negate, NoAttrs{}, {value}});
    auto root = graph.add({OpKind::Negate, NoAttrs{}, {inner}});
    eggc::run(graph, compileRules({restrict ? restricted : original}));
    EXPECT_EQ(graph.find(root) == graph.find(value), !restrict);
  }
}
TEST_F(SemanticRewriteTest, CallbackFailureNeverInsertsChildren) {
  using namespace patterns;
  using namespace tensorlang::ops;
  const auto x = tensor_var("x");
  const auto lhs = transpose(x, attributes<TransposeAttrs>{Axes{1, 0}});
  for (int mode = 0; mode < 5; ++mode) {
    auto definition =
        rule("callback", lhs).build([=](const Match& m, RhsBuilder& rhs) {
          auto value = rhs.make(negate, rhs.ref(m[x]));
          if (mode == 0) return rhs.reject("unsupported dimensions");
          if (mode == 1)
            return rhs.make(transpose, TransposeAttrs{{0, 0}}, value);
          if (mode == 2) return value;  // Root shape differs.
          if (mode == 3) return rhs.ref(m[tensor_var("missing")]);
          return rhs.make(transpose,
                          TransposeAttrs{m[attribute_var<Axes>("missing")]},
                          value);
        });
    TensorEGraph graph;
    auto value = input(graph, 0, integer({2, 3}));
    graph.add({OpKind::Transpose, TransposeAttrs{{1, 0}}, {value}});
    auto count = graph.node_count();
    auto options = reporting();
    eggc::run(graph,
              compileRules({definition}, NumericalPolicy::PreserveEvaluation,
                           options));
    EXPECT_EQ(graph.node_count(), count);
    EXPECT_GT(options.report->at("callback").rejections.size(), 0);
  }
}
TEST_F(SemanticRewriteTest, CallbackRechecksCapturedOperatorProperties) {
  using namespace patterns;
  using namespace tensorlang::ops;
  const auto x = tensor_var("x");
  const auto F = operator_var("F", 1);
  const auto dynamic_type = integer({mlir::ShapedType::kDynamic});
  // Broadcast inference can accept a dynamic input dimension. Its layout
  // linearity contract requires static operands even when the output is static.
  auto definition = rule("capture", F(x))
                        .when(linear_in(F, 0))
                        .build([=](const Match&, RhsBuilder& rhs) {
                          auto dynamic = rhs.make(tensorlang::ops::input,
                                                  InputAttrs{99, dynamic_type});
                          return rhs.apply(F, {dynamic});
                        });
  TensorEGraph graph;
  auto value = input(graph, 0, integer({3}));
  graph.add(
      {OpKind::BroadcastInDim, BroadcastAttrs{{1}, integer({2, 3})}, {value}});
  auto count = graph.node_count();
  auto options = reporting();
  eggc::run(graph, compileRules({definition},
                                NumericalPolicy::PreserveEvaluation, options));
  EXPECT_EQ(graph.node_count(), count);
  EXPECT_GT(options.report->at("capture").rejections.count(
                "RHS: layout algebra requires static shapes"),
            0);
}
TEST_F(SemanticRewriteTest, CallbackRebuildsCapturedOperator) {
  using namespace patterns;
  using namespace tensorlang::ops;
  const auto x = tensor_var("x"), y = tensor_var("y");
  const auto F = operator_var("F", 2);
  auto definition = rule("callback-swap", F(x, y))
                        .when(commutative(F))
                        .build([=](const Match& m, RhsBuilder& rhs) {
                          return rhs.apply(F, {rhs.ref(m[y]), rhs.ref(m[x])});
                        });
  TensorEGraph graph;
  auto a = input(graph, 0, integer()), b = input(graph, 1, integer());
  auto root = graph.add({OpKind::Add, NoAttrs{}, {a, b}});
  eggc::run(graph, compileRules({definition}));
  auto expected = graph.add({OpKind::Add, NoAttrs{}, {b, a}});
  EXPECT_EQ(graph.find(root), graph.find(expected));
}
}  // namespace

}  // namespace joint_shard

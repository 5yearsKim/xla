#include <array>
#include <memory>
#include <random>
#include <stdexcept>

#include "mlir/IR/MLIRContext.h"
#include "eggc/runner.hpp"
#include "gtest/gtest.h"
#include "research/joint_shard/bridge/tensor_lang/tensor_rewrites.h"

namespace joint_shard {

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
      if (attrs.lhs_contracting != std::vector<int64_t>{1} ||
          attrs.rhs_contracting != std::vector<int64_t>{0} ||
          !attrs.lhs_batching.empty() || !attrs.rhs_batching.empty() ||
          a.columns != b.rows)
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
TEST_F(SemanticRewriteTest, RejectsTyposArityAndUnboundOperatorsAtLoad) {
  for (auto source : {"rule bad { typo(?x) => ?x where Involution(F); }",
                      "rule bad { add(?x) => ?x where Involution(F); }",
                      "rule bad { F(?x) => G(?x) where Involution(F); }",
                      "rule bad { F(?x) => F(?x, ?x) where Involution(F); }",
                      "rule bad { F(?x) => ?y where Involution(F); }",
                      "rule bad { F(?x) => ?x where LinearIn(F, 1); }",
                      "rule bad { F(?) => ? where Involution(F); }",
                      "rule bad { scale(?x) => ?x where Scalar(?x); }",
                      "rule bad { F(?x) => ?x where Scalar(?y); }",
                      "rule bad { F(?x) => ?x where Scalar(F); }",
                      "rule bad { F(?x) => ?x where Scalar(?x, 0); }",
                      "rule bad { F(?x) => ?x where Uniform(?x, 0); }",
                      "rule dup { F(?x) => ?x where Involution(F); } rule dup "
                      "{ F(?x) => ?x where Involution(F); }"}) {
    EXPECT_THROW(parseSemanticRules(source), std::invalid_argument) << source;
  }
}
TEST_F(SemanticRewriteTest, SourceNameAppearsInDiagnostics) {
  auto options = reporting();
  options.source_name = "my.rules";
  try {
    parseSemanticRules("rule broken {", NumericalPolicy::PreserveEvaluation,
                       options);
    FAIL();
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what()).find("my.rules:"), std::string::npos);
  }
}
TEST_F(SemanticRewriteTest, ConjunctionUsesSharedProperties) {
  TensorEGraph graph;
  auto x = input(graph, 0, integer()), y = input(graph, 1, integer());
  auto root = graph.add({OpKind::Add, NoAttrs{}, {x, y}});
  auto rules = parseSemanticRules(
      "rule swap { F(?x, ?y) => F(?y, ?x) where Elementwise(F) and "
      "Commutative(F); }");
  eggc::run(graph, rules);
  auto reverse = graph.add({OpKind::Add, NoAttrs{}, {y, x}});
  EXPECT_EQ(graph.find(root), graph.find(reverse));
}
TEST_F(SemanticRewriteTest, StrictFloatingReorderingIsRejectedWithReason) {
  TensorEGraph graph;
  auto type =
      mlir::RankedTensorType::get({2, 3}, mlir::Float32Type::get(&context));
  auto x = input(graph, 0, type), y = input(graph, 1, type);
  graph.add({OpKind::Add, NoAttrs{}, {x, y}});
  auto options = reporting();
  auto count = graph.node_count();
  eggc::run(graph,
            parseSemanticRules(
                "rule swap { F(?x, ?y) => F(?y, ?x) where Commutative(F); }",
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
  TensorEGraph graph;
  auto x = input(graph, 0, integer({2, 3}));
  auto f = graph.add({OpKind::Transpose, TransposeAttrs{{1, 0}}, {x}});
  graph.add({OpKind::Transpose, TransposeAttrs{{1, 0}}, {f}});
  auto options = reporting();
  auto count = graph.node_count();
  // negate(?x) would insert a node before the add discovered its shape mismatch
  // in the old recursive instantiator. Preparation rejects the entire RHS.
  auto rules = parseSemanticRules(
      "rule mismatch { F(F(?x)) => add(negate(?x), F(?x)) where Involution(F); "
      "}",
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
      graph, parseSemanticRules(defaultSemanticRules(),
                                NumericalPolicy::PreserveEvaluation, options));
  EXPECT_EQ(result.reason, eggc::StopReason::SearchLimit);
  EXPECT_GT(options.report->at("commute-binary-operator").budget_stops, 0);
}
TEST_F(SemanticRewriteTest, DotLinearityFactorsTwoDots) {
  TensorEGraph graph;
  auto a = input(graph, 0, integer({2, 3})),
       b = input(graph, 1, integer({3, 4}));
  auto c = input(graph, 2, integer({3, 4}));
  DotGeneralAttrs attrs{{1}, {0}, {}, {}, {}, {}, integer({2, 4}), {}};
  auto ab = graph.add({OpKind::DotGeneral, attrs, {a, b}});
  auto ac = graph.add({OpKind::DotGeneral, attrs, {a, c}});
  auto root = graph.add({OpKind::Add, NoAttrs{}, {ab, ac}});
  graph.rebuild();
  auto original = TensorExtractor(graph).find_best(root).second;
  auto rules = parseSemanticRules(
      "rule factor { add(F(?a, ?b), F(?a, ?c)) => F(?a, add(?b, ?c)) where "
      "LinearIn(F, 1); }");
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
  TensorEGraph graph;
  auto x = input(graph, 0, integer({2, 2})),
       scale = input(graph, 1, integer({2, 2}));
  auto mul = graph.add({OpKind::Multiply, NoAttrs{}, {scale, x}});
  graph.add({OpKind::Transpose, TransposeAttrs{{1, 0}}, {mul}});
  auto options = reporting();
  auto count = graph.node_count();
  auto rules = parseSemanticRules(
      "rule scale { F(multiply(?scale, ?x)) => multiply(?scale, F(?x)) where "
      "HomogeneousIn(F, 0) and Uniform(?scale); }",
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
    auto rules = parseSemanticRules(
        "rule move { F(scale(?s, ?x)) => scale(?s, F(?x)) "
        "where HomogeneousIn(F, 0) and Scalar(?s); }");
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
    DotGeneralAttrs attrs{{1}, {0}, {}, {}, {}, {}, integer({2, 4}), {}};
    auto root = graph.add({OpKind::DotGeneral, attrs,
                           slot == 0 ? std::vector<eggc::Id>{scaled, b}
                                     : std::vector<eggc::Id>{a, scaled}});
    graph.rebuild();
    auto original = TensorExtractor(graph).find_best(root).second;
    auto rules = parseSemanticRules(defaultSemanticRules());
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
  auto rules = parseSemanticRules(
      "rule scalar { F(F(?coefficient)) => ?coefficient "
      "where Involution(F) and Scalar(?coefficient); }");
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
  TensorEGraph graph;
  auto x = input(graph, 0, integer({2, 3}));
  auto tensor_scale = input(graph, 1, integer({2, 3}));
  auto multiply = graph.add({OpKind::Multiply, NoAttrs{}, {tensor_scale, x}});
  graph.add({OpKind::Transpose, TransposeAttrs{{1, 0}}, {multiply}});
  auto options = reporting();
  auto rules = parseSemanticRules(
      "rule move { F(scale(?s, ?x)) => scale(?s, F(?x)) "
      "where HomogeneousIn(F, 0) and Scalar(?s); }",
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
  auto limited = parseSemanticRules(
      "rule bounded { scale(?s, ?x) => scale(?s, ?x) where Scalar(?s); }",
      NumericalPolicy::PreserveEvaluation, options);
  auto result = eggc::run(bounded, limited);
  EXPECT_EQ(result.reason, eggc::StopReason::SearchLimit);
  EXPECT_GT(options.report->at("bounded").budget_stops, 0);
}
TEST_F(SemanticRewriteTest, StrictFloatScaleDoesNotReorderMultiplyOperands) {
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
    auto rules = parseSemanticRules(
        "rule move { F(scale(?s, ?x)) => scale(?s, F(?x)) "
        "where HomogeneousIn(F, 0) and Scalar(?s); }",
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
  auto rules = parseSemanticRules(
      "rule move { F(scale(?s, ?x)) => scale(?s, F(?x)) "
      "where HomogeneousIn(F, 0) and Scalar(?s); }");
  eggc::run(graph, rules);
  auto transpose = graph.add({OpKind::Transpose, TransposeAttrs{{1, 0}}, {x}});
  auto broadcast = graph.add(
      {OpKind::BroadcastInDim, BroadcastAttrs{{}, integer({3, 2})}, {scalar}});
  auto expected =
      graph.add({OpKind::Multiply, NoAttrs{}, {broadcast, transpose}});
  EXPECT_EQ(graph.find(root), graph.find(expected));
}
TEST_F(SemanticRewriteTest, UniformPredicateAcceptsBroadcastOfScalar) {
  TensorEGraph graph;
  auto scalar = input(graph, 0, integer({}));
  auto uniform = graph.add(
      {OpKind::BroadcastInDim, BroadcastAttrs{{}, integer({2, 3})}, {scalar}});
  auto negate = graph.add({OpKind::Negate, NoAttrs{}, {uniform}});
  auto root = graph.add({OpKind::Negate, NoAttrs{}, {negate}});
  auto rules = parseSemanticRules(
      "rule uniform { F(F(?value)) => ?value "
      "where Involution(F) and Uniform(?value); }");
  eggc::run(graph, rules);
  EXPECT_EQ(graph.find(root), graph.find(uniform));
}
TEST_F(SemanticRewriteTest, ScaleOnRankZeroNeedsNoBroadcast) {
  TensorEGraph graph;
  auto s = input(graph, 0, integer({}));
  auto x = input(graph, 1, integer({}));
  auto product = graph.add({OpKind::Multiply, NoAttrs{}, {s, x}});
  auto root = graph.add({OpKind::Negate, NoAttrs{}, {product}});
  auto rules = parseSemanticRules(
      "rule move { F(scale(?s, ?x)) => scale(?s, F(?x)) "
      "where HomogeneousIn(F, 0) and Scalar(?s); }");
  eggc::run(graph, rules);
  auto negated = graph.add({OpKind::Negate, NoAttrs{}, {x}});
  auto expected = graph.add({OpKind::Multiply, NoAttrs{}, {s, negated}});
  EXPECT_EQ(graph.find(root), graph.find(expected));
  EXPECT_TRUE(graph.classes_for_op(OpKind::BroadcastInDim).empty());
}
TEST_F(SemanticRewriteTest, NonScalarRhsScaleLeavesGraphUnchanged) {
  TensorEGraph graph;
  auto a = input(graph, 0, integer({2, 3}));
  auto b = input(graph, 1, integer({2, 3}));
  graph.add({OpKind::Add, NoAttrs{}, {a, b}});
  auto options = reporting();
  auto rules = parseSemanticRules(
      "rule bad { F(?s, ?x) => scale(?s, negate(?x)) "
      "where Elementwise(F); }",
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
}  // namespace

}  // namespace joint_shard

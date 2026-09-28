#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

#include "mlir/IR/MLIRContext.h"
#include "eggc/runner.hpp"
#include "gtest/gtest.h"
#include "research/joint_shard/bridge/tensor_lang/tensor_patterns.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace joint_shard {
using patterns::attributes;
namespace {
// Independent dense interpreter: enumerate operand elements rather than reuse
// the optimizer's inference or matrix-chain transformations.
template <class T>
struct Dense {
  std::vector<int64_t> shape;
  std::vector<T> values;
};
std::vector<int64_t> coordinates(size_t index,
                                 const std::vector<int64_t>& shape) {
  std::vector<int64_t> result(shape.size());
  for (size_t i = shape.size(); i-- > 0;) {
    result[i] = index % shape[i];
    index /= shape[i];
  }
  return result;
}
size_t offset(const std::vector<int64_t>& coordinates,
              const std::vector<int64_t>& shape) {
  size_t result = 0;
  for (size_t i = 0; i < shape.size(); ++i)
    result = result * shape[i] + coordinates[i];
  return result;
}
bool contains(const std::vector<int64_t>& axes, int64_t axis) {
  return std::find(axes.begin(), axes.end(), axis) != axes.end();
}
template <class T>
Dense<T> evaluate(const TensorRecExpr& expression,
                  const std::vector<Dense<T>>& inputs) {
  std::vector<Dense<T>> values;
  for (const auto& node : expression.nodes) {
    if (node.op == OpKind::Constant) {
      auto attr = llvm::cast<mlir::DenseFPElementsAttr>(
          std::get<ConstantAttrs>(node.attrs).value);
      auto shape = llvm::cast<mlir::RankedTensorType>(attr.getType());
      Dense<T> result{{shape.getShape().begin(), shape.getShape().end()}, {}};
      for (auto value : attr.getValues<llvm::APFloat>())
        result.values.push_back(static_cast<T>(value.convertToDouble()));
      values.push_back(std::move(result));
      continue;
    }
    if (node.op == OpKind::Input) {
      values.push_back(inputs.at(std::get<InputAttrs>(node.attrs).index));
      continue;
    }
    const auto& lhs = values.at(node.operands[0]);
    if (node.op == OpKind::Reshape) {
      auto type = std::get<ReshapeAttrs>(node.attrs).result_type;
      values.push_back(
          {{type.getShape().begin(), type.getShape().end()}, lhs.values});
      continue;
    }
    if (node.op == OpKind::BroadcastInDim || node.op == OpKind::Transpose) {
      Dense<T> result;
      std::vector<int64_t> axes;
      if (node.op == OpKind::BroadcastInDim) {
        const auto& attrs = std::get<BroadcastAttrs>(node.attrs);
        axes = attrs.dimensions;
        result.shape.assign(attrs.result_type.getShape().begin(),
                            attrs.result_type.getShape().end());
      } else {
        axes = std::get<TransposeAttrs>(node.attrs).permutation;
        for (auto axis : axes) result.shape.push_back(lhs.shape[axis]);
      }
      size_t count = std::accumulate(result.shape.begin(), result.shape.end(),
                                     size_t{1}, std::multiplies<>());
      for (size_t i = 0; i < count; ++i) {
        const auto position = coordinates(i, result.shape);
        std::vector<int64_t> source(lhs.shape.size());
        for (size_t axis = 0; axis < axes.size(); ++axis) {
          if (node.op == OpKind::Transpose)
            source[axes[axis]] = position[axis];
          else
            source[axis] = lhs.shape[axis] == 1 ? 0 : position[axes[axis]];
        }
        result.values.push_back(lhs.values[offset(source, lhs.shape)]);
      }
      values.push_back(std::move(result));
      continue;
    }
    if (node.op == OpKind::Multiply || node.op == OpKind::Divide ||
        node.op == OpKind::Subtract) {
      const auto& rhs = values.at(node.operands[1]);
      Dense<T> result{lhs.shape, {}};
      for (size_t i = 0; i < lhs.values.size(); ++i)
        result.values.push_back(
            node.op == OpKind::Multiply ? lhs.values[i] * rhs.values[i]
            : node.op == OpKind::Divide ? lhs.values[i] / rhs.values[i]
                                        : lhs.values[i] - rhs.values[i]);
      values.push_back(std::move(result));
      continue;
    }
    if (node.op == OpKind::Reduce) {
      const auto& attrs = std::get<ReduceAttrs>(node.attrs);
      if (attrs.kind != ReduceKind::Sum)
        throw std::logic_error("oracle requires sum");
      Dense<T> result;
      for (size_t i = 0; i < lhs.shape.size(); ++i)
        if (!contains(attrs.axes, i)) result.shape.push_back(lhs.shape[i]);
      size_t count = std::accumulate(result.shape.begin(), result.shape.end(),
                                     size_t{1}, std::multiplies<>());
      result.values.resize(count);
      for (size_t i = 0; i < lhs.values.size(); ++i) {
        auto position = coordinates(i, lhs.shape);
        std::vector<int64_t> output;
        for (size_t axis = 0; axis < position.size(); ++axis)
          if (!contains(attrs.axes, axis)) output.push_back(position[axis]);
        result.values[offset(output, result.shape)] += lhs.values[i];
      }
      values.push_back(std::move(result));
      continue;
    }
    if (node.op != OpKind::DotGeneral)
      throw std::logic_error("unsupported oracle operation");
    const auto& rhs = values.at(node.operands[1]);
    const auto& attrs = std::get<DotGeneralAttrs>(node.attrs);
    Dense<T> result;
    for (auto axis : attrs.dimensions.lhs_batching)
      result.shape.push_back(lhs.shape[axis]);
    for (size_t axis = 0; axis < lhs.shape.size(); ++axis)
      if (!contains(attrs.dimensions.lhs_batching, axis) &&
          !contains(attrs.dimensions.lhs_contracting, axis))
        result.shape.push_back(lhs.shape[axis]);
    for (size_t axis = 0; axis < rhs.shape.size(); ++axis)
      if (!contains(attrs.dimensions.rhs_batching, axis) &&
          !contains(attrs.dimensions.rhs_contracting, axis))
        result.shape.push_back(rhs.shape[axis]);
    size_t count = std::accumulate(result.shape.begin(), result.shape.end(),
                                   size_t{1}, std::multiplies<>());
    result.values.resize(count);
    for (size_t i = 0; i < lhs.values.size(); ++i) {
      auto l = coordinates(i, lhs.shape);
      for (size_t j = 0; j < rhs.values.size(); ++j) {
        auto r = coordinates(j, rhs.shape);
        bool matches = true;
        for (size_t k = 0; k < attrs.dimensions.lhs_contracting.size(); ++k)
          matches &= l[attrs.dimensions.lhs_contracting[k]] ==
                     r[attrs.dimensions.rhs_contracting[k]];
        for (size_t k = 0; k < attrs.dimensions.lhs_batching.size(); ++k)
          matches &= l[attrs.dimensions.lhs_batching[k]] ==
                     r[attrs.dimensions.rhs_batching[k]];
        if (!matches) continue;
        std::vector<int64_t> output;
        for (auto axis : attrs.dimensions.lhs_batching)
          output.push_back(l[axis]);
        for (size_t axis = 0; axis < l.size(); ++axis)
          if (!contains(attrs.dimensions.lhs_batching, axis) &&
              !contains(attrs.dimensions.lhs_contracting, axis))
            output.push_back(l[axis]);
        for (size_t axis = 0; axis < r.size(); ++axis)
          if (!contains(attrs.dimensions.rhs_batching, axis) &&
              !contains(attrs.dimensions.rhs_contracting, axis))
            output.push_back(r[axis]);
        result.values[offset(output, result.shape)] +=
            lhs.values[i] * rhs.values[j];
      }
    }
    values.push_back(std::move(result));
  }
  return values.back();
}

class SemanticAttributeRewriteTest : public ::testing::Test {
 protected:
  mlir::MLIRContext context;
  SemanticRuleOptions options;
  void SetUp() override {
    context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();
    options.report =
        std::make_shared<std::map<std::string, SemanticRuleStats>>();
  }
  mlir::RankedTensorType type(std::vector<int64_t> shape,
                              bool floating = false) {
    mlir::Type element = floating
                             ? mlir::Type(mlir::Float32Type::get(&context))
                             : mlir::Type(mlir::IntegerType::get(&context, 32));
    return mlir::RankedTensorType::get(shape, element);
  }
  mlir::ArrayAttr precision(mlir::stablehlo::Precision value) {
    auto attr = mlir::stablehlo::PrecisionAttr::get(&context, value);
    return mlir::ArrayAttr::get(&context, {attr, attr});
  }
  eggc::Id input(TensorRecExpr& expr, unsigned index,
                 std::vector<int64_t> shape, bool floating) {
    return expr.add(
        {OpKind::Input, InputAttrs{index, type(shape, floating)}, {}});
  }
  eggc::Id dot(TensorRecExpr& expr, eggc::Id a, eggc::Id b,
               std::vector<int64_t> shape, int64_t lc, int64_t rc, bool batched,
               bool floating) {
    DotGeneralAttrs attrs{
        DotDimensions{
            {lc},
            {rc},
            batched ? std::vector<int64_t>{0} : std::vector<int64_t>{},
            batched ? std::vector<int64_t>{0} : std::vector<int64_t>{}},
        {},
        {},
        type(shape, floating),
        {}};
    return expr.add({OpKind::DotGeneral, attrs, {a, b}});
  }
  eggc::Id sum(TensorRecExpr& expr, eggc::Id x, int64_t axis, bool floating) {
    return expr.add(
        {OpKind::Reduce,
         ReduceAttrs{ReduceKind::Sum,
                     {axis},
                     canonicalReductionInitializer(
                         ReduceKind::Sum, type({}, floating).getElementType())},
         {x}});
  }
  TensorRecExpr chain(int form, bool right, bool floating = false) {
    TensorRecExpr expr;
    bool batch = form != 0;
    auto a = input(
        expr, 0,
        batch ? std::vector<int64_t>{2, 3, 4} : std::vector<int64_t>{2, 3},
        floating);
    auto b = input(expr, 1,
                   form == 0   ? std::vector<int64_t>{3, 5}
                   : form == 3 ? std::vector<int64_t>{4, 5}
                   : form == 4 ? std::vector<int64_t>{2, 5, 4}
                               : std::vector<int64_t>{2, 4, 5},
                   floating);
    auto c = input(expr, 2,
                   form == 1 || form == 4 ? std::vector<int64_t>{2, 5, 7}
                                          : std::vector<int64_t>{5, 7},
                   floating);
    auto final =
        batch ? std::vector<int64_t>{2, 3, 7} : std::vector<int64_t>{2, 7};
    if (!right) {
      auto ab = dot(
          expr, a, b,
          batch ? std::vector<int64_t>{2, 3, 5} : std::vector<int64_t>{2, 5},
          batch ? 2 : 1,
          form == 0 || form == 3 ? 0
          : form == 4            ? 2
                                 : 1,
          form == 1 || form == 2 || form == 4, floating);
      dot(expr, ab, c, final, batch ? 2 : 1, form == 1 || form == 4 ? 1 : 0,
          form == 1 || form == 4, floating);
    } else {
      auto bc =
          dot(expr, b, c,
              form == 0   ? std::vector<int64_t>{3, 7}
              : form == 3 ? std::vector<int64_t>{4, 7}
                          : std::vector<int64_t>{2, 4, 7},
              form == 0 || form == 3 || form == 4 ? 1 : 2,
              form == 1 || form == 4 ? 1 : 0, form == 1 || form == 4, floating);
      dot(expr, a, bc, final, batch ? 2 : 1, form == 0 || form == 3 ? 0 : 1,
          form == 1 || form == 2 || form == 4, floating);
    }
    return expr;
  }
  TensorRecExpr interchange(bool denominator, bool right,
                            bool floating = false) {
    TensorRecExpr expr;
    auto x = input(expr, 0,
                   denominator ? std::vector<int64_t>{2, 3, 4}
                               : std::vector<int64_t>{2, 5, 3},
                   floating);
    auto w = input(expr, 1,
                   denominator ? std::vector<int64_t>{2, 5, 4}
                               : std::vector<int64_t>{3, 7},
                   floating);
    if (denominator) {
      if (!right) {
        auto kernel = dot(expr, x, w, {2, 3, 5}, 2, 2, true, floating);
        sum(expr, kernel, 2, floating);
      } else {
        auto reduced = sum(expr, w, 1, floating);
        dot(expr, x, reduced, {2, 3}, 2, 1, true, floating);
      }
    } else if (!right) {
      auto reduced = sum(expr, x, 1, floating);
      dot(expr, reduced, w, {2, 7}, 1, 0, false, floating);
    } else {
      auto projected = dot(expr, x, w, {2, 5, 7}, 2, 0, false, floating);
      sum(expr, projected, 1, floating);
    }
    return expr;
  }
  void equivalent(const TensorRecExpr& source, const TensorRecExpr& target,
                  NumericalPolicy policy) {
    TensorEGraph graph;
    auto root = graph.add_expr(source);
    eggc::run(graph, buildSemanticRules(policy, options));
    auto count = graph.node_count();
    auto other = graph.add_expr(target);
    EXPECT_EQ(graph.node_count(), count) << "RHS was not emitted by a rewrite";
    EXPECT_EQ(graph.find(root), graph.find(other));
    EXPECT_EQ(tensorFacts(graph, root).type, tensorFacts(graph, other).type);
  }
  template <class T>
  void oracle(const TensorRecExpr& source, const TensorRecExpr& target,
              bool positive_denominator = false) {
    std::mt19937 random(42);
    std::vector<Dense<T>> inputs;
    for (const auto& node : source.nodes) {
      if (node.op != OpKind::Input) continue;
      auto tensor = std::get<InputAttrs>(node.attrs).type;
      Dense<T> input{{tensor.getShape().begin(), tensor.getShape().end()}, {}};
      for (int64_t i = 0; i < tensor.getNumElements(); ++i) {
        if constexpr (std::is_same_v<T, uint32_t>)
          input.values.push_back(random());
        else
          input.values.push_back(
              static_cast<float>(static_cast<int>(random() % 17) - 8) / 7);
      }
      // Denominators for division tests are finite, positive and well-scaled.
      if constexpr (std::is_floating_point_v<T>)
        if (positive_denominator && std::get<InputAttrs>(node.attrs).index == 2)
          for (auto& value : input.values) value = T{0.5} + std::abs(value);
      inputs.push_back(std::move(input));
    }
    auto a = evaluate(source, inputs), b = evaluate(target, inputs);
    EXPECT_EQ(a.shape, b.shape);
    ASSERT_EQ(a.values.size(), b.values.size());
    for (size_t i = 0; i < a.values.size(); ++i) {
      if constexpr (std::is_same_v<T, uint32_t>)
        EXPECT_EQ(a.values[i], b.values[i]);
      else
        EXPECT_NEAR(a.values[i], b.values[i],
                    2e-5 * (1 + std::abs(a.values[i])));
    }
  }
};

// Build original and expected expressions independently of the rule callbacks.
// Forms cover row, scalar, per-query, batched and singleton contraction maps.
TensorRecExpr normalizedDot(mlir::MLIRContext& context, int form, bool nested,
                            bool after, bool floating = true,
                            bool varying = false) {
  const bool batched = form >= 3;
  const std::vector<int64_t> xshape =
      batched ? std::vector<int64_t>{2, 3, 4} : std::vector<int64_t>{3, 4};
  const std::vector<int64_t> vshape =
      batched ? std::vector<int64_t>{2, 4, 5} : std::vector<int64_t>{4, 5};
  const std::vector<int64_t> outshape =
      batched ? std::vector<int64_t>{2, 3, 5} : std::vector<int64_t>{3, 5};
  const std::vector<std::vector<int64_t>> shapes{
      {3}, {3, 1}, {}, {2, 3}, {2, 3, 1}, {3}, {1, 3, 1}, {}};
  const std::vector<std::vector<int64_t>> maps{
      {0}, {0, 1}, {}, {0, 1}, {0, 1, 2}, {1}, {0, 1, 2}, {}};
  auto dshape = varying ? xshape : shapes.at(form);
  auto axes = maps.at(form);
  if (varying) {
    axes.resize(xshape.size());
    std::iota(axes.begin(), axes.end(), 0);
  }
  auto element = floating ? mlir::Type(mlir::Float32Type::get(&context))
                          : mlir::Type(mlir::IntegerType::get(&context, 32));
  auto type = [&](const std::vector<int64_t>& shape) {
    return mlir::RankedTensorType::get(shape, element);
  };
  TensorRecExpr expr;
  auto x = expr.add({OpKind::Input, InputAttrs{0, type(xshape)}, {}}),
       v = expr.add({OpKind::Input, InputAttrs{1, type(vshape)}, {}}),
       d = expr.add({OpKind::Input, InputAttrs{2, type(dshape)}, {}});
  auto make_dot = [&](eggc::Id lhs) {
    return expr.add(
        {OpKind::DotGeneral,
         DotGeneralAttrs{
             DotDimensions{
                 {batched ? 2 : 1},
                 {batched ? 1 : 0},
                 batched ? std::vector<int64_t>{0} : std::vector<int64_t>{},
                 batched ? std::vector<int64_t>{0} : std::vector<int64_t>{}},
             {},
             {},
             type(outshape),
             {}},
         {lhs, v}});
  };
  if (after) {
    // The test's input forms only map a singleton contraction as the last axis.
    if (!axes.empty() &&
        axes.back() == static_cast<int64_t>(xshape.size() - 1)) {
      dshape.pop_back();
      axes.pop_back();
      d = expr.add({OpKind::Reshape, ReshapeAttrs{type(dshape)}, {d}});
    }
    auto projected = make_dot(x);
    auto broadcast = expr.add(
        {OpKind::BroadcastInDim, BroadcastAttrs{axes, type(outshape)}, {d}});
    expr.add({OpKind::Divide, NoAttrs{}, {projected, broadcast}});
  } else {
    if (nested) {
      auto intermediate = xshape;
      if (!varying) intermediate.back() = 1;
      d = expr.add({OpKind::BroadcastInDim,
                    BroadcastAttrs{axes, type(intermediate)},
                    {d}});
      axes.resize(xshape.size());
      std::iota(axes.begin(), axes.end(), 0);
    }
    auto broadcast = expr.add(
        {OpKind::BroadcastInDim, BroadcastAttrs{axes, type(xshape)}, {d}});
    auto divided = expr.add({OpKind::Divide, NoAttrs{}, {x, broadcast}});
    make_dot(divided);
  }
  return expr;
}

TEST_F(SemanticAttributeRewriteTest,
       DotDivisionMapsDirectAndNestedDenominatorsNumerically) {
  for (int form = 0; form < 8; ++form)
    for (bool nested : {false, true}) {
      SCOPED_TRACE(form);
      SCOPED_TRACE(nested);
      auto source = normalizedDot(context, form, nested, false),
           target = normalizedDot(context, form, nested, true);
      // Preserve an explicitly typed precision config as well as absence.
      if (form % 2)
        for (auto* expression : {&source, &target})
          for (auto& node : expression->nodes)
            if (node.op == OpKind::DotGeneral)
              std::get<DotGeneralAttrs>(node.attrs).precision_config =
                  precision(mlir::stablehlo::Precision::DEFAULT);
      equivalent(source, target, NumericalPolicy::AllowReassociation);
      oracle<float>(source, target, true);
      const auto name =
          nested ? "dot-divide-broadcast-nested" : "dot-divide-broadcast";
      EXPECT_GT(options.report->at(name).applied, 0);
    }
}

TEST_F(SemanticAttributeRewriteTest, DotDivisionRangeChangesRequirePermission) {
  const auto source = normalizedDot(context, 0, false, false),
             target = normalizedDot(context, 0, false, true);
  const std::vector<Dense<float>> inputs{
      {{3, 4}, std::vector<float>(12, 1e20f)},
      {{4, 5}, std::vector<float>(20, 1e20f)},
      {{3}, std::vector<float>(3, 1e20f)}};
  auto before = evaluate(source, inputs), after = evaluate(target, inputs);
  for (size_t i = 0; i < before.values.size(); ++i) {
    EXPECT_TRUE(std::isfinite(before.values[i]));
    EXPECT_TRUE(std::isinf(after.values[i]));
  }
  TensorEGraph graph;
  graph.add_expr(source);
  const auto count = graph.node_count();
  options.permissions =
      numericalPermissions(NumericalPolicy::AllowReassociation);
  options.permissions->rewrite_dot_division = false;
  eggc::run(graph,
            buildSemanticRules(NumericalPolicy::AllowReassociation, options));
  EXPECT_EQ(graph.node_count(), count);
}

TEST_F(SemanticAttributeRewriteTest,
       DotDivisionRejectsVaryingDenominatorsIntegersAndStrictPolicy) {
  for (int mode = 0; mode < 4; ++mode) {
    TensorEGraph graph;
    graph.add_expr(
        normalizedDot(context, 4, true, false, mode != 1, mode == 0));
    auto rules =
        buildSemanticRules(mode == 2 ? NumericalPolicy::PreserveEvaluation
                                     : NumericalPolicy::AllowReassociation,
                           options);
    if (mode == 3) {
      options.permissions =
          numericalPermissions(NumericalPolicy::AllowReassociation);
      options.permissions->rewrite_dot_division = false;
      rules = buildSemanticRules(NumericalPolicy::AllowReassociation, options);
    }
    const auto count = graph.node_count();
    eggc::run(graph, rules);
    EXPECT_EQ(graph.node_count(), count);
    for (auto name : {"dot-divide-broadcast", "dot-divide-broadcast-nested"})
      EXPECT_EQ(options.report->at(name).applied, 0);
  }
}

TEST_F(SemanticAttributeRewriteTest,
       DotDivisionRejectsUnsupportedDimensionsAndPrecision) {
  for (bool nondefault_precision : {false, true}) {
    auto source = normalizedDot(context, 3, false, false);
    auto& attrs = std::get<DotGeneralAttrs>(source.nodes.back().attrs);
    if (nondefault_precision) {
      attrs.precision_config = precision(mlir::stablehlo::Precision::HIGH);
    } else {
      // A valid batch dot with transposed lhs axes, outside our narrow scope.
      attrs.dimensions.lhs_contracting = {1};
      attrs.result_type = type({2, 4, 5}, true);
      for (auto& node : source.nodes)
        if (node.op == OpKind::Input &&
            std::get<InputAttrs>(node.attrs).index == 1)
          node.attrs = InputAttrs{1, type({2, 3, 5}, true)};
    }
    TensorEGraph graph;
    graph.add_expr(source);
    auto count = graph.node_count();
    eggc::run(graph,
              buildSemanticRules(NumericalPolicy::AllowReassociation, options));
    EXPECT_EQ(graph.node_count(), count);
    EXPECT_EQ(options.report->at("dot-divide-broadcast").applied, 0);
  }
}

TEST_F(SemanticAttributeRewriteTest,
       GramNormUsesSampleMatricesForRectangularFeatureShapes) {
  for (bool floating : {false, true})
    for (bool folded : {false, true}) {
      auto expression = [&](bool sample) {
        TensorRecExpr expr;
        auto a = input(expr, 0, {3, 5}, floating),
             b = input(expr, 1, {3, 7}, floating);
        eggc::Id left, right;
        if (sample) {
          left = dot(expr, a, a, {3, 3}, 1, 1, false, floating);
          right = dot(expr, b, b, {3, 3}, 1, 1, false, floating);
        } else {
          auto lhs =
              folded
                  ? a
                  : expr.add({OpKind::Transpose, TransposeAttrs{{1, 0}}, {a}});
          left = right =
              dot(expr, lhs, b, {5, 7}, folded ? 0 : 1, 0, false, floating);
        }
        auto product = expr.add({OpKind::Multiply, NoAttrs{}, {left, right}});
        expr.add({OpKind::Reduce,
                  ReduceAttrs{ReduceKind::Sum,
                              {0, 1},
                              canonicalReductionInitializer(
                                  ReduceKind::Sum,
                                  type({}, floating).getElementType())},
                  {product}});
        return expr;
      };
      const auto source = expression(false), target = expression(true);
      equivalent(source, target,
                 floating ? NumericalPolicy::AllowReassociation
                          : NumericalPolicy::PreserveEvaluation);
      if (floating) {
        oracle<float>(source, target);
        TensorEGraph graph;
        graph.add_expr(source);
        const auto count = graph.node_count();
        eggc::run(graph, buildSemanticRules(NumericalPolicy::PreserveEvaluation,
                                            options));
        EXPECT_EQ(graph.node_count(), count);
      } else {
        oracle<uint32_t>(source, target);
      }
    }
}

TEST_F(SemanticAttributeRewriteTest, BroadcastAttributesMatchAndBuildExactly) {
  using namespace patterns;
  using namespace tensorlang::ops;
  const auto x = tensor_var("x");
  const auto attrs = attribute_var<BroadcastAttrs>("attrs");
  auto source = broadcast_in_dim(negate(x), attrs);
  auto target = negate(broadcast_in_dim(x, attrs));
  auto rules = compileRules({rule("move-negate", source, target)});
  TensorEGraph graph;
  auto value = graph.add({OpKind::Input, InputAttrs{0, type({3})}, {}});
  auto negated = graph.add({OpKind::Negate, NoAttrs{}, {value}});
  BroadcastAttrs broadcast{{0}, type({3, 5})};
  auto root = graph.add({OpKind::BroadcastInDim, broadcast, {negated}});
  eggc::run(graph, rules);
  auto expected = graph.add({OpKind::BroadcastInDim, broadcast, {value}});
  expected = graph.add({OpKind::Negate, NoAttrs{}, {expected}});
  EXPECT_EQ(graph.find(root), graph.find(expected));
  // A literal captures the target type too; matching only dimensions is wrong.
  options.report->clear();
  rules = compileRules(
      {rule("literal",
            broadcast_in_dim(negate(x), BroadcastAttrs{{0}, type({3, 4})}),
            negate(broadcast_in_dim(x, BroadcastAttrs{{0}, type({3, 4})})))},
      NumericalPolicy::PreserveEvaluation, options);
  eggc::run(graph, rules);
  EXPECT_EQ(options.report->at("literal").structural_matches, 0);
}

TEST_F(SemanticAttributeRewriteTest, AggressiveRawMomentsMatchCanonicalMeans) {
  for (auto axes : {std::vector<int64_t>{0}, std::vector<int64_t>{1},
                    std::vector<int64_t>{0, 1}})
    for (bool keepdims : {false, true}) {
      auto expression = [&](bool raw, bool wrong_count = false) {
        TensorRecExpr expr;
        auto x = input(expr, 0, {2, 4}, true);
        int size = 1;
        std::vector<int64_t> shape{2, 4}, kept = shape, surviving,
                                          reduced_shape;
        for (auto axis : axes) {
          size *= shape[axis];
          kept[axis] = 1;
        }
        for (int64_t axis = 0; axis < 2; ++axis)
          if (!contains(axes, axis)) {
            surviving.push_back(axis);
            reduced_shape.push_back(shape[axis]);
          }
        auto n = expr.add(
            {OpKind::Constant,
             ConstantAttrs{mlir::DenseElementsAttr::get(
                 type({}, true), static_cast<float>(size + wrong_count))},
             {}});
        auto reduction = [&](eggc::Id value) {
          return expr.add({OpKind::Reduce,
                           ReduceAttrs{ReduceKind::Sum, axes,
                                       canonicalReductionInitializer(
                                           ReduceKind::Sum,
                                           type({}, true).getElementType())},
                           {value}});
        };
        auto sums = reduction(x);
        if (raw) {
          auto squared = expr.add({OpKind::Multiply, NoAttrs{}, {x, x}});
          auto sum_squared = reduction(squared);
          auto squared_sum =
              expr.add({OpKind::Multiply, NoAttrs{}, {sums, sums}});
          auto divisor =
              expr.add({OpKind::BroadcastInDim,
                        BroadcastAttrs{{}, type(reduced_shape, true)},
                        {n}});
          auto correction =
              expr.add({OpKind::Divide, NoAttrs{}, {squared_sum, divisor}});
          expr.add({OpKind::Subtract, NoAttrs{}, {sum_squared, correction}});
        } else {
          if (keepdims)
            sums = expr.add({OpKind::BroadcastInDim,
                             BroadcastAttrs{surviving, type(kept, true)},
                             {sums}});
          auto divisor = expr.add(
              {OpKind::BroadcastInDim,
               BroadcastAttrs{{}, type(keepdims ? kept : reduced_shape, true)},
               {n}});
          auto mean = expr.add({OpKind::Divide, NoAttrs{}, {sums, divisor}});
          auto expanded = expr.add(
              {OpKind::BroadcastInDim,
               BroadcastAttrs{keepdims ? std::vector<int64_t>{0, 1} : surviving,
                              type(shape, true)},
               {mean}});
          auto centered =
              expr.add({OpKind::Subtract, NoAttrs{}, {x, expanded}});
          auto squared =
              expr.add({OpKind::Multiply, NoAttrs{}, {centered, centered}});
          reduction(squared);
        }
        return expr;
      };
      options.permissions =
          numericalPermissions(NumericalPolicy::AllowReassociation);
      options.permissions->rewrite_raw_moments = true;
      auto original = expression(false), target = expression(true);
      equivalent(original, target, NumericalPolicy::AllowReassociation);
      oracle<float>(original, target);
      for (int mode = 0; mode < 3; ++mode) {
        auto permissions = numericalPermissions(
            mode == 0 ? NumericalPolicy::PreserveEvaluation
                      : NumericalPolicy::AllowReassociation);
        permissions.rewrite_raw_moments = mode == 2;
        options.permissions = permissions;
        TensorEGraph graph;
        graph.add_expr(expression(false, mode == 2));
        auto count = graph.node_count();
        eggc::run(graph, buildSemanticRules(NumericalPolicy::AllowReassociation,
                                            options));
        EXPECT_EQ(graph.node_count(), count);
      }
    }
}

TEST_F(SemanticAttributeRewriteTest,
       MatrixAndBatchedChainsHaveNumericallyEquivalentAlternatives) {
  for (int form = 0; form < 5; ++form) {
    SCOPED_TRACE(form);
    for (bool reverse : {false, true}) {
      auto original = chain(form, reverse), rewritten = chain(form, !reverse);
      equivalent(original, rewritten, NumericalPolicy::PreserveEvaluation);
      oracle<uint32_t>(original, rewritten);
      auto a = chain(form, reverse, true), b = chain(form, !reverse, true);
      equivalent(a, b, NumericalPolicy::AllowReassociation);
      oracle<float>(a, b);
    }
  }
}
TEST_F(SemanticAttributeRewriteTest, SumDotAndKernelDenominatorBothDirections) {
  for (bool denominator : {false, true})
    for (bool reverse : {false, true}) {
      auto a = interchange(denominator, reverse),
           b = interchange(denominator, !reverse);
      equivalent(a, b, NumericalPolicy::PreserveEvaluation);
      oracle<uint32_t>(a, b);
      auto af = interchange(denominator, reverse, true),
           bf = interchange(denominator, !reverse, true);
      equivalent(af, bf, NumericalPolicy::AllowReassociation);
      oracle<float>(af, bf);
    }
}
TEST_F(SemanticAttributeRewriteTest,
       AttributeVariablesAndLiteralsMatchDistinctPermutations) {
  const auto pattern_x = patterns::tensor_var("x");
  const auto pattern_p = patterns::attribute_var<Axes>("p");

  auto rules = compileRules(
      {patterns::rule("cancel",
                      tensorlang::ops::transpose(
                          tensorlang::ops::transpose(
                              pattern_x, attributes<TransposeAttrs>{pattern_p}),
                          attributes<TransposeAttrs>{pattern_p}),
                      pattern_x)
           .when({patterns::rank(pattern_x, 2)})},
      NumericalPolicy::PreserveEvaluation, options);
  for (bool equal : {false, true}) {
    TensorEGraph graph;
    auto x = graph.add({OpKind::Input, InputAttrs{0, type({3, 3})}, {}});
    auto inner = graph.add({OpKind::Transpose, TransposeAttrs{{1, 0}}, {x}});
    auto root = graph.add({OpKind::Transpose,
                           TransposeAttrs{equal ? std::vector<int64_t>{1, 0}
                                                : std::vector<int64_t>{0, 1}},
                           {inner}});
    eggc::run(graph, rules);
    EXPECT_EQ(graph.find(root) == graph.find(x), equal);
  }
}
TEST_F(SemanticAttributeRewriteTest,
       StrictAndPartialPermissionsRejectWithoutMutation) {
  for (int mask = 0; mask < 32; ++mask) {
    TensorEGraph graph;
    graph.add_expr(chain(0, false, true));
    auto count = graph.node_count();
    NumericalPermissions permissions;
    permissions.rewrite_dot_arithmetic = mask & 1;
    permissions.reassociate_floating_point = mask & 2;
    permissions.distribute_floating_point = mask & 4;
    permissions.assume_finite = mask & 8;
    permissions.ignore_signed_zero = mask & 16;
    options.permissions = permissions;
    eggc::run(graph,
              buildSemanticRules(NumericalPolicy::PreserveEvaluation, options));
    if (mask != 31)
      EXPECT_EQ(graph.node_count(), count) << mask;
    else
      EXPECT_GT(graph.node_count(), count);
  }
}
TEST_F(SemanticAttributeRewriteTest,
       PrecisionIsPreservedAndNondefaultIsRejected) {
  for (bool high : {false, true}) {
    auto source = chain(0, false, true), target = chain(0, true, true);
    auto config = precision(high ? mlir::stablehlo::Precision::HIGH
                                 : mlir::stablehlo::Precision::DEFAULT);
    for (auto* expr : {&source, &target})
      for (auto& node : expr->nodes)
        if (node.op == OpKind::DotGeneral)
          std::get<DotGeneralAttrs>(node.attrs).precision_config = config;
    if (!high)
      equivalent(source, target, NumericalPolicy::AllowReassociation);
    else {
      TensorEGraph graph;
      graph.add_expr(source);
      auto count = graph.node_count();
      eggc::run(graph, buildSemanticRules(NumericalPolicy::AllowReassociation,
                                          options));
      EXPECT_EQ(graph.node_count(), count);
      EXPECT_GT(
          options.report->at("reassociate-matrix-right").rejections.size(), 0);
    }
  }
}
TEST_F(SemanticAttributeRewriteTest,
       IncompatiblePrecisionAlgorithmAndMetadataDoNotMatch) {
  for (int mode = 0; mode < 3; ++mode) {
    TensorEGraph graph;
    auto expr = chain(0, false);
    auto& inner = std::get<DotGeneralAttrs>(expr.nodes[3].attrs);
    if (mode == 0)
      inner.precision_config = precision(mlir::stablehlo::Precision::DEFAULT);
    if (mode == 1)
      inner.algorithm = mlir::StringAttr::get(&context, "unsupported");
    if (mode == 2)
      inner.extra_attributes = mlir::DictionaryAttr::get(
          &context,
          {mlir::NamedAttribute(mlir::StringAttr::get(&context, "hint"),
                                mlir::UnitAttr::get(&context))});
    graph.add_expr(expr);
    auto count = graph.node_count();
    eggc::run(graph,
              buildSemanticRules(NumericalPolicy::AllowReassociation, options));
    EXPECT_EQ(graph.node_count(), count);
  }
}
TEST_F(SemanticAttributeRewriteTest,
       InvalidRhsAndRootTypeLeaveNoSpeculativeNodes) {
  const auto pattern_x = patterns::tensor_var("x");

  const auto xvar = pattern_x;
  const auto lhs =
      tensorlang::ops::transpose(xvar, attributes<TransposeAttrs>{Axes{1, 0}});
  for (auto rhs :
       {tensorlang::ops::transpose(tensorlang::ops::negate(xvar),
                                   attributes<TransposeAttrs>{Axes{0, 0}}),
        tensorlang::ops::reduce(
            tensorlang::ops::negate(xvar),
            attributes<ReduceAttrs>{ReduceKind::Sum, Axes{0}})}) {
    TensorEGraph graph;
    auto x = graph.add({OpKind::Input, InputAttrs{0, type({2, 3})}, {}});
    graph.add({OpKind::Transpose, TransposeAttrs{{1, 0}}, {x}});
    auto count = graph.node_count();
    eggc::run(
        graph,
        compileRules(
            {patterns::rule("bad", lhs, rhs).when(patterns::rank(xvar, 2))},
            NumericalPolicy::PreserveEvaluation, options));
    EXPECT_EQ(graph.node_count(), count);
    EXPECT_GT(options.report->at("bad").rejections.size(), 0);
  }
}
TEST_F(SemanticAttributeRewriteTest,
       RuleGroupsFilterDedicatedArithmeticGuards) {
  options.enable_associativity = false;
  options.enable_linearity = false;
  options.enable_homogeneity = false;
  auto rules = buildSemanticRules(NumericalPolicy::AllowReassociation, options);
  for (const auto& rule : rules) {
    EXPECT_EQ(rule.name.find("reassociate-"), std::string::npos);
    EXPECT_EQ(rule.name.find("sequence-"), std::string::npos);
    EXPECT_EQ(rule.name.find("kernel-denominator-"), std::string::npos);
    EXPECT_EQ(rule.name.find("kernel-attention-summaries"), std::string::npos);
    EXPECT_EQ(rule.name.find("lora-projection"), std::string::npos);
    EXPECT_EQ(rule.name.find("dot-divide-broadcast"), std::string::npos);
    EXPECT_EQ(rule.name.find("feature-gram-to-sample-gram"), std::string::npos);
  }
}
TEST_F(SemanticAttributeRewriteTest,
       ConcreteAttributeSearchHonorsLimitsAndCancellation) {
  TensorEGraph graph;
  graph.add_expr(chain(0, false));
  auto count = graph.node_count();
  options.visit_limit = 1;
  auto run = eggc::run(
      graph, buildSemanticRules(NumericalPolicy::PreserveEvaluation, options));
  EXPECT_EQ(run.reason, eggc::StopReason::SearchLimit);
  EXPECT_EQ(graph.node_count(), count);
  options.visit_limit = 100000;
  auto rules = buildSemanticRules(NumericalPolicy::PreserveEvaluation, options);
  for (const auto& rule : rules)
    EXPECT_FALSE(rule.custom_search(
        graph, [](const auto&) { return true; }, [] { return true; }));
}
TEST_F(SemanticAttributeRewriteTest,
       CanonicalInitializerConstructionCoversReducersAndDtypes) {
  for (auto element : {mlir::Type(mlir::IntegerType::get(&context, 32)),
                       mlir::Type(mlir::IntegerType::get(
                           &context, 32, mlir::IntegerType::Unsigned)),
                       mlir::Type(mlir::Float16Type::get(&context)),
                       mlir::Type(mlir::Float32Type::get(&context))})
    for (auto kind : {ReduceKind::Sum, ReduceKind::Product, ReduceKind::Min,
                      ReduceKind::Max}) {
      auto initializer = canonicalReductionInitializer(kind, element);
      EXPECT_TRUE(
          canonicalReductionIdentity({kind, {0}, initializer}, element));
    }
}
TEST_F(SemanticAttributeRewriteTest, WholeRecordCapturePreservesDotMetadata) {
  using namespace patterns;
  using namespace tensorlang::ops;
  const auto a = tensor_var("a"), b = tensor_var("b");
  const auto metadata = attribute_var<DotGeneralAttrs>("metadata");
  const auto dot_pattern = dot_general(a, b, metadata);
  auto declarative =
      rule("cancel-around-dot", negate(negate(dot_pattern)), dot_pattern);
  auto callback =
      rule("cancel-around-dot-callback", negate(negate(dot_pattern)))
          .build([=](const Match& m, RhsBuilder& rhs) {
            return rhs.make(dot_general, m[metadata], rhs.ref(m[a]),
                            rhs.ref(m[b]));
          });
  DotGeneralAttrs attrs{
      dot_dims({1}, {0}), precision(mlir::stablehlo::Precision::DEFAULT),
      mlir::StringAttr::get(&context, "opaque-algorithm"), type({3, 5}),
      mlir::DictionaryAttr::get(
          &context, {mlir::NamedAttribute(
                        mlir::StringAttr::get(&context, "producer"),
                        mlir::StringAttr::get(&context, "opaque-metadata"))})};
  for (const auto& definition : {declarative, callback}) {
    TensorEGraph graph;
    auto lhs = graph.add({OpKind::Input, InputAttrs{0, type({3, 4})}, {}});
    auto rhs = graph.add({OpKind::Input, InputAttrs{1, type({4, 5})}, {}});
    auto dot = graph.add({OpKind::DotGeneral, attrs, {lhs, rhs}});
    auto inner = graph.add({OpKind::Negate, NoAttrs{}, {dot}});
    auto root = graph.add({OpKind::Negate, NoAttrs{}, {inner}});
    auto count = graph.node_count();
    eggc::run(graph, compileRules({definition}));
    EXPECT_EQ(graph.find(root), graph.find(dot));
    EXPECT_EQ(graph.node_count(), count);
    EXPECT_EQ(graph.find(root),
              graph.find(graph.add({OpKind::DotGeneral, attrs, {lhs, rhs}})));
  }
}

TEST_F(SemanticAttributeRewriteTest,
       SchemaFieldsSupportNewAttributeTypesInBothBuilders) {
  using namespace patterns;
  using namespace tensorlang::ops;
  const auto x = tensor_var("x");
  // RankedTensorType wasn't in the old pattern attribute variant. Reshape
  // wasn't supported by its typed concrete pattern constructors either.
  const auto shape = attribute_var<mlir::RankedTensorType>("shape");
  const auto lhs = reshape(negate(x), attributes<ReshapeAttrs>{shape});
  auto declarative = rule("reshape-negate", lhs,
                          negate(reshape(x, attributes<ReshapeAttrs>{shape})));
  auto callback =
      rule("reshape-negate-callback", lhs)
          .build([=](const Match& m, RhsBuilder& rhs) {
            return rhs.make(negate, rhs.make(reshape, ReshapeAttrs{m[shape]},
                                             rhs.ref(m[x])));
          });
  for (const auto& definition : {declarative, callback}) {
    TensorEGraph graph;
    auto value = graph.add({OpKind::Input, InputAttrs{0, type({2, 3})}, {}});
    auto negated = graph.add({OpKind::Negate, NoAttrs{}, {value}});
    auto root =
        graph.add({OpKind::Reshape, ReshapeAttrs{type({6})}, {negated}});
    eggc::run(graph, compileRules({definition}));
    auto shaped =
        graph.add({OpKind::Reshape, ReshapeAttrs{type({6})}, {value}});
    auto expected = graph.add({OpKind::Negate, NoAttrs{}, {shaped}});
    EXPECT_EQ(graph.find(root), graph.find(expected));
  }
}

TEST_F(SemanticAttributeRewriteTest,
       WildcardsMatchButCannotConstructRhsAttributes) {
  using namespace patterns;
  using namespace tensorlang::ops;
  const auto x = tensor_var("x");
  const auto lhs = reshape(negate(x), attributes<ReshapeAttrs>{any_attribute});
  const auto target_type = type({6});
  auto definition =
      rule("wildcard-shape", lhs).build([=](const Match& m, RhsBuilder& rhs) {
        auto shape = m.type(x);
        if (!shape || !shape.hasStaticShape() || shape.getNumElements() != 6)
          return rhs.reject("unexpected shape");
        return rhs.make(negate, rhs.make(reshape, ReshapeAttrs{target_type},
                                         rhs.ref(m[x])));
      });
  TensorEGraph graph;
  auto value = graph.add({OpKind::Input, InputAttrs{0, type({2, 3})}, {}});
  auto negated = graph.add({OpKind::Negate, NoAttrs{}, {value}});
  auto root = graph.add({OpKind::Reshape, ReshapeAttrs{type({6})}, {negated}});
  eggc::run(graph, compileRules({definition}));
  auto shaped = graph.add({OpKind::Reshape, ReshapeAttrs{type({6})}, {value}});
  auto expected = graph.add({OpKind::Negate, NoAttrs{}, {shaped}});
  EXPECT_EQ(graph.find(root), graph.find(expected));
  EXPECT_THROW(compileRules({rule("bad-wildcard", negate(x), lhs)}),
               std::invalid_argument);
  EXPECT_THROW(reshape(x), std::invalid_argument);
}

TEST_F(SemanticAttributeRewriteTest, CppAttributeBindingsHaveConsistentTypes) {
  using namespace patterns;
  using namespace tensorlang::ops;
  const auto x = tensor_var("x");
  const auto p = attribute_var<Axes>("p");
  EXPECT_THROW(
      compileRules({rule(
          "unbound", transpose(x, attributes<TransposeAttrs>{p}),
          transpose(x, attributes<TransposeAttrs>{attribute_var<Axes>("q")}))}),
      std::invalid_argument);
  EXPECT_THROW(
      compileRules({rule(
          "collision",
          transpose(x, attributes<TransposeAttrs>{attribute_var<Axes>("x")}),
          x)}),
      std::invalid_argument);
  EXPECT_THROW(
      compileRules({rule(
          "wrong-type",
          reduce(x, attributes<ReduceAttrs>{attribute_var<ReduceKind>("p"), p}),
          x)}),
      std::invalid_argument);
}
TEST_F(SemanticAttributeRewriteTest, CallbackComputesPermutationComposition) {
  using namespace patterns;
  using namespace tensorlang::ops;
  const auto x = tensor_var("x");
  const auto inner = attribute_var<Axes>("inner"),
             outer = attribute_var<Axes>("outer");
  auto definition =
      rule("compose", transpose(transpose(x, attributes<TransposeAttrs>{inner}),
                                attributes<TransposeAttrs>{outer}))
          .build([=](const Match& m, RhsBuilder& rhs) {
            auto p = m[inner], q = m[outer];
            if (p.size() != q.size())
              return rhs.reject("permutation ranks differ");
            Axes composed;
            for (auto axis : q) composed.push_back(p.at(axis));
            return rhs.make(transpose, TransposeAttrs{std::move(composed)},
                            rhs.ref(m[x]));
          });
  TensorEGraph graph;
  auto value = graph.add({OpKind::Input, InputAttrs{0, type({2, 3, 4})}, {}});
  auto first =
      graph.add({OpKind::Transpose, TransposeAttrs{{1, 2, 0}}, {value}});
  auto root =
      graph.add({OpKind::Transpose, TransposeAttrs{{1, 0, 2}}, {first}});
  eggc::run(graph, compileRules({definition}));
  auto expected =
      graph.add({OpKind::Transpose, TransposeAttrs{{2, 1, 0}}, {value}});
  EXPECT_EQ(graph.find(root), graph.find(expected));
}
TEST_F(SemanticAttributeRewriteTest,
       CallbackCapturesWholeDotDimensionsAndPrecision) {
  using namespace patterns;
  using namespace tensorlang::ops;
  const auto a = tensor_var("a"), b = tensor_var("b"), c = tensor_var("c");
  const auto dims = attribute_var<DotDimensions>("dims");
  const auto p = attribute_var<mlir::ArrayAttr>("p");
  auto definition =
      rule("callback-reassociate",
           dot_general(dot_general(a, b, attributes<DotGeneralAttrs>{dims, p}),
                       c, attributes<DotGeneralAttrs>{dims, p}))
          .when({rank(a, 2), rank(b, 2), rank(c, 2), dot_reassociation()})
          .build([=](const Match& m, RhsBuilder& rhs) {
            if (m[dims] != dot_dims({1}, {0}))
              return rhs.reject("requires matrix dimensions");
            auto bc = rhs.make(dot_general, DotGeneralAttrs{m[dims], m[p]},
                               rhs.ref(m[b]), rhs.ref(m[c]));
            return rhs.make(dot_general, DotGeneralAttrs{m[dims], m[p]},
                            rhs.ref(m[a]), bc);
          });
  for (bool floating : {false, true}) {
    for (auto policy : {NumericalPolicy::PreserveEvaluation,
                        NumericalPolicy::AllowReassociation}) {
      TensorEGraph graph;
      auto original = chain(0, false, floating);
      auto config = precision(mlir::stablehlo::Precision::DEFAULT);
      for (auto& node : original.nodes)
        if (node.op == OpKind::DotGeneral)
          std::get<DotGeneralAttrs>(node.attrs).precision_config = config;
      auto root = graph.add_expr(original);
      auto count = graph.node_count();
      eggc::run(graph, compileRules({definition}, policy, options));
      if (floating && policy == NumericalPolicy::PreserveEvaluation) {
        EXPECT_EQ(graph.node_count(), count);
        continue;
      }
      auto expected = chain(0, true, floating);
      for (auto& node : expected.nodes)
        if (node.op == OpKind::DotGeneral)
          std::get<DotGeneralAttrs>(node.attrs).precision_config = config;
      count = graph.node_count();
      auto other = graph.add_expr(expected);
      EXPECT_EQ(graph.node_count(), count);
      EXPECT_EQ(graph.find(root), graph.find(other));
      if (floating)
        oracle<float>(original, expected);
      else
        oracle<uint32_t>(original, expected);
    }
  }
}
TEST_F(SemanticAttributeRewriteTest, CallbackAttributeTypeMismatchIsRejected) {
  using namespace patterns;
  using namespace tensorlang::ops;
  const auto x = tensor_var("x");
  const auto p = attribute_var<Axes>("p");
  auto definition =
      rule("bad-callback-type", transpose(x, attributes<TransposeAttrs>{p}))
          .build([=](const Match& m, RhsBuilder& rhs) {
            auto dimensions = m[attribute_var<DotDimensions>("p")];
            return rhs.make(dot_general, DotGeneralAttrs{dimensions},
                            rhs.ref(m[x]), rhs.ref(m[x]));
          });
  TensorEGraph graph;
  auto value = graph.add({OpKind::Input, InputAttrs{0, type({2, 3})}, {}});
  graph.add({OpKind::Transpose, TransposeAttrs{{1, 0}}, {value}});
  auto count = graph.node_count();
  eggc::run(graph, compileRules({definition},
                                NumericalPolicy::PreserveEvaluation, options));
  EXPECT_EQ(graph.node_count(), count);
  EXPECT_GT(options.report->at("bad-callback-type").rejections.size(), 0);
}
TEST_F(SemanticAttributeRewriteTest,
       AdditionalSumDotShapesHaveValidAlternatives) {
  struct Case {
    std::vector<int64_t> x_shape, w_shape, full_shape, result_shape;
    std::vector<int64_t> input_axes, output_axes;
    int reduced_lc, reduced_rc, full_lc, full_rc;
    bool reduce_right;
  };
  const Case cases[] = {
      {{3, 4}, {4, 5}, {3, 5}, {5}, {0}, {0}, 0, 0, 1, 0, false},
      {{2, 3, 2, 4},
       {4, 5},
       {2, 3, 2, 5},
       {2, 5},
       {1, 2},
       {1, 2},
       1,
       0,
       3,
       0,
       false},
      {{5, 4}, {3, 4}, {3, 5}, {3}, {0}, {1}, 1, 0, 1, 1, true},
      {{2, 4, 5}, {3, 4}, {3, 2, 5}, {3, 5}, {0}, {1}, 1, 0, 1, 1, true},
  };
  for (const auto& c : cases) {
    for (bool floating : {false, true}) {
      auto expression = [&](bool sum_first) {
        TensorRecExpr expr;
        auto x = input(expr, 0, c.x_shape, floating),
             w = input(expr, 1, c.w_shape, floating);
        const auto sum_axes = [&](eggc::Id value,
                                  const std::vector<int64_t>& axes) {
          return expr.add(
              {OpKind::Reduce,
               ReduceAttrs{
                   ReduceKind::Sum, axes,
                   canonicalReductionInitializer(
                       ReduceKind::Sum, type({}, floating).getElementType())},
               {value}});
        };
        if (sum_first) {
          auto reduced = sum_axes(x, c.input_axes);
          dot(expr, c.reduce_right ? w : reduced, c.reduce_right ? reduced : w,
              c.result_shape, c.reduced_lc, c.reduced_rc, false, floating);
        } else {
          auto projected =
              dot(expr, c.reduce_right ? w : x, c.reduce_right ? x : w,
                  c.full_shape, c.full_lc, c.full_rc, false, floating);
          sum_axes(projected, c.output_axes);
        }
        return expr;
      };
      auto first = expression(true), second = expression(false);
      auto policy = floating ? NumericalPolicy::AllowReassociation
                             : NumericalPolicy::PreserveEvaluation;
      equivalent(first, second, policy);
      equivalent(second, first, policy);
      if (floating)
        oracle<float>(first, second);
      else
        oracle<uint32_t>(first, second);
    }
  }
}
}  // namespace
}  // namespace joint_shard

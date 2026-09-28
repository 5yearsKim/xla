#include <array>
#include <limits>
#include <numeric>

#include "llvm/Support/Casting.h"
#include "research/joint_shard/bridge/tensor_lang/tensor_patterns.h"

namespace joint_shard {
using patterns::attributes;
namespace {
// Limited to ordinary matrix and batch-matrix products. Their surviving lhs
// axes retain their positions in the result, so no general dimension solver
// or output permutation is needed.
patterns::Expression divideAfterDot(
    const patterns::Match& m, patterns::RhsBuilder& rhs, patterns::TensorVar x,
    patterns::TensorVar v, patterns::TensorVar denominator,
    patterns::AttributeVar<DotDimensions> dimensions,
    patterns::AttributeVar<mlir::ArrayAttr> precision,
    Axes broadcast_dimensions) {
  using namespace patterns;
  using namespace tensorlang::ops;
  auto xt = m.type(x), vt = m.type(v), dt = m.type(denominator);
  if (!xt || !vt || !dt || !xt.hasStaticShape() || !vt.hasStaticShape() ||
      !dt.hasStaticShape())
    return rhs.reject("dot division requires static tensor shapes");
  const auto dims = m[dimensions];
  if (!((xt.getRank() == 2 && vt.getRank() == 2 &&
         dims == dot_dims({1}, {0})) ||
        (xt.getRank() == 3 && vt.getRank() == 3 &&
         dims == dot_dims({2}, {1}, {0}, {0}))))
    return rhs.reject("dot division supports matrix and batch-matrix layouts");
  if (broadcast_dimensions.size() != static_cast<size_t>(dt.getRank()))
    return rhs.reject("dot division broadcast rank mismatch");

  Axes kept_shape, output_dimensions;
  const int64_t contracted = xt.getRank() - 1;
  for (int64_t i = 0; i < dt.getRank(); ++i) {
    if (broadcast_dimensions[i] == contracted) {
      if (dt.getDimSize(i) != 1)
        return rhs.reject("denominator varies over the contracted dimension");
      // A singleton mapped onto the contracted dimension can be dropped.
    } else {
      kept_shape.push_back(dt.getDimSize(i));
      output_dimensions.push_back(broadcast_dimensions[i]);
    }
  }
  auto d = rhs.ref(m[denominator]);
  if (kept_shape.size() != static_cast<size_t>(dt.getRank()))
    d = rhs.make(reshape,
                 ReshapeAttrs{mlir::RankedTensorType::get(kept_shape,
                                                          dt.getElementType())},
                 d);
  Axes result_shape(xt.getShape().begin(), xt.getShape().end());
  result_shape.back() = vt.getDimSize(vt.getRank() - 1);
  auto result_type =
      mlir::RankedTensorType::get(result_shape, xt.getElementType());
  return rhs.make(divide,
                  rhs.make(dot_general, DotGeneralAttrs{dims, m[precision]},
                           rhs.ref(m[x]), rhs.ref(m[v])),
                  rhs.make(broadcast_in_dim,
                           BroadcastAttrs{output_dimensions, result_type}, d));
}
}  // namespace

std::vector<TensorRewrite> buildSemanticRules(NumericalPolicy policy,
                                              SemanticRuleOptions options) {
  using namespace patterns;
  using namespace tensorlang::ops;
  const auto x = tensor_var("x"), y = tensor_var("y"), z = tensor_var("z"),
             a = tensor_var("a"), b = tensor_var("b"), c = tensor_var("c"),
             s = tensor_var("s"), w = tensor_var("w"), q = tensor_var("q"),
             k = tensor_var("k");
  // The number is the operand count (arity): F(x, y) is binary; U(x) is unary.
  const auto F = operator_var("F", 2), U = operator_var("F", 1);
  std::vector<RuleDefinition> rules{
      rule("commute-binary-operator", F(x, y), F(y, x)).when(commutative(F)),
      rule("associate-binary-operator-right", F(F(x, y), z), F(x, F(y, z)))
          .when(associative(F)),
      rule("associate-binary-operator-left", F(x, F(y, z)), F(F(x, y), z))
          .when(associative(F)),
      rule("eliminate-double-involution", U(U(x)), x).when(involution(U)),
      rule("distribute-left-linear-operator-over-add", F(add(x, y), a),
           add(F(x, a), F(y, a)))
          .when(linear_in(F, 0)),
      rule("distribute-right-linear-operator-over-add", F(a, add(x, y)),
           add(F(a, x), F(a, y)))
          .when(linear_in(F, 1)),
      rule("factor-left-linear-operator-from-add", add(F(x, a), F(y, a)),
           F(add(x, y), a))
          .when(linear_in(F, 0)),
      rule("factor-right-linear-operator-from-add", add(F(a, x), F(a, y)),
           F(a, add(x, y)))
          .when(linear_in(F, 1)),
      rule("move-scaling-out-of-left-operand", F(scale(s, x), a),
           scale(s, F(x, a)))
          .when({homogeneous_in(F, 0), scalar(s)}),
      rule("move-scaling-out-of-right-operand", F(a, scale(s, x)),
           scale(s, F(a, x)))
          .when({homogeneous_in(F, 1), scalar(s)}),
      rule("distribute-unary-linear-operator-over-add", U(add(x, y)),
           add(U(x), U(y)))
          .when(linear_in(U, 0)),
      rule("factor-unary-linear-operator-from-add", add(U(x), U(y)),
           U(add(x, y)))
          .when(linear_in(U, 0)),
      rule("move-uniform-scaling-out-of-unary-operator", U(scale(s, x)),
           scale(s, U(x)))
          .when({homogeneous_in(U, 0), scalar(s)}),
  };

  // These signatures encode dimension lineage explicitly. They share one C++
  // rule construction; more general lineage can be supplied via .build().
  const auto matrix = dot_dims({1}, {0});
  const auto batch = dot_dims({2}, {1}, {0}, {0});
  const auto shared = dot_dims({2}, {0});
  struct Reassociation {
    const char* name;
    std::array<unsigned, 3> ranks;
    DotDimensions left_inner, left_outer, right_outer, right_inner;
  };
  const std::array cases{
      Reassociation{"matrix", {2, 2, 2}, matrix, matrix, matrix, matrix},
      Reassociation{"batched", {3, 3, 3}, batch, batch, batch, batch},
      Reassociation{
          "batched-shared-weight", {3, 3, 2}, batch, shared, batch, shared},
      Reassociation{
          "shared-weights", {3, 2, 2}, shared, shared, shared, matrix},
      Reassociation{"kernel-attention",
                    {3, 3, 3},
                    dot_dims({2}, {2}, {0}, {0}),
                    batch,
                    batch,
                    dot_dims({1}, {1}, {0}, {0})},
  };
  const auto precision = attribute_var<mlir::ArrayAttr>("precision");
  for (const auto& signature : cases) {
    const auto left = dot_general(
        dot_general(
            a, b, attributes<DotGeneralAttrs>{signature.left_inner, precision}),
        c, attributes<DotGeneralAttrs>{signature.left_outer, precision});
    const auto right = dot_general(
        a,
        dot_general(
            b, c,
            attributes<DotGeneralAttrs>{signature.right_inner, precision}),
        attributes<DotGeneralAttrs>{signature.right_outer, precision});
    const auto guards = {rank(a, signature.ranks[0]),
                         rank(b, signature.ranks[1]),
                         rank(c, signature.ranks[2]), dot_reassociation()};
    const std::string name = std::string("reassociate-") + signature.name;
    rules.push_back(rule(name + "-right", left, right).when(guards));
    rules.push_back(rule(name + "-left", right, left).when(guards));
  }
  // Explicit signatures keep dimension changes reviewable. All variants use
  // the same equation and preserve precision, sum identities and inferred
  // types.
  struct SumDot {
    const char* project_name;
    const char* reduce_name;
    unsigned input_rank, weight_rank;
    bool reduce_right;
    Axes input_axes, output_axes;
    DotDimensions reduced_dot, full_dot;
  };
  const std::array sums{
      SumDot{"project-before-sequence-sum",
             "sum-before-sequence-projection",
             3,
             2,
             false,
             {1},
             {1},
             matrix,
             shared},
      SumDot{"project-before-row-sum",
             "sum-before-row-projection",
             2,
             2,
             false,
             {0},
             {0},
             dot_dims({0}, {0}),
             matrix},
      SumDot{"project-before-spatial-sum",
             "sum-before-spatial-projection",
             4,
             2,
             false,
             {1, 2},
             {1, 2},
             matrix,
             dot_dims({3}, {0})},
      SumDot{"kernel-denominator-after-dot",
             "kernel-denominator-before-dot",
             3,
             3,
             true,
             {1},
             {2},
             batch,
             dot_dims({2}, {2}, {0}, {0})},
      SumDot{"kernel-denominator-transposed-after-dot",
             "kernel-denominator-transposed-before-dot",
             3,
             3,
             true,
             {2},
             {2},
             batch,
             batch},
      SumDot{"kernel-denominator-matrix-after-dot",
             "kernel-denominator-matrix-before-dot",
             2,
             2,
             true,
             {0},
             {1},
             matrix,
             dot_dims({1}, {1})},
      SumDot{"project-before-weight-sum",
             "sum-weights-before-projection",
             3,
             2,
             true,
             {0},
             {1},
             matrix,
             dot_dims({1}, {1})},
  };
  for (const auto& signature : sums) {
    const auto reduced = reduce(
        x, attributes<ReduceAttrs>{ReduceKind::Sum, signature.input_axes});
    const auto sum_first =
        signature.reduce_right
            ? dot_general(
                  w, reduced,
                  attributes<DotGeneralAttrs>{signature.reduced_dot, precision})
            : dot_general(reduced, w,
                          attributes<DotGeneralAttrs>{signature.reduced_dot,
                                                      precision});
    const auto full =
        signature.reduce_right
            ? dot_general(
                  w, x,
                  attributes<DotGeneralAttrs>{signature.full_dot, precision})
            : dot_general(
                  x, w,
                  attributes<DotGeneralAttrs>{signature.full_dot, precision});
    const auto dot_first = reduce(
        full, attributes<ReduceAttrs>{ReduceKind::Sum, signature.output_axes});
    const auto guards = {rank(x, signature.input_rank),
                         rank(w, signature.weight_rank), sum_dot_interchange()};
    rules.push_back(
        rule(signature.project_name, sum_first, dot_first).when(guards));
    rules.push_back(
        rule(signature.reduce_name, dot_first, sum_first).when(guards));
  }

  // Discover the complete low-rank path in one match rather than relying on
  // a distribution pass followed by reassociation of its newly emitted dot.
  const auto effective_weight = dot_general(
      x,
      add(w, dot_general(a, b, attributes<DotGeneralAttrs>{matrix, precision})),
      attributes<DotGeneralAttrs>{matrix, precision});
  const auto low_rank =
      add(dot_general(x, w, attributes<DotGeneralAttrs>{matrix, precision}),
          dot_general(
              dot_general(x, a, attributes<DotGeneralAttrs>{matrix, precision}),
              b, attributes<DotGeneralAttrs>{matrix, precision}));
  const auto lora_guards = {rank(x, 2), rank(w, 2), rank(a, 2), rank(b, 2),
                            dot_arithmetic()};
  rules.push_back(rule("expand-lora-projection", effective_weight, low_rank)
                      .when(lora_guards));
  rules.push_back(rule("factor-lora-projection", low_rank, effective_weight)
                      .when(lora_guards));

  // Coordinate numerator and denominator so a score-sized intermediate can
  // disappear from both consumers. The feature map remains an opaque input;
  // this equation never moves through softmax or substitutes a kernel for it.
  const auto v = tensor_var("v");
  const auto D0 = operator_var("D0", 1), D1 = operator_var("D1", 1);
  struct KernelAttention {
    const char* suffix;
    DotDimensions score_dims, summary_dims;
    Axes key_sum_axes;
  };
  const std::array kernels{
      KernelAttention{
          "", dot_dims({2}, {2}, {0}, {0}), dot_dims({1}, {1}, {0}, {0}), {1}},
      KernelAttention{"-transposed-key", batch, batch, {2}},
  };
  const auto kernel_guards = {rank(q, 3), rank(k, 3), rank(v, 3),
                              dot_arithmetic()};
  for (const auto& signature : kernels) {
    const auto scores = dot_general(
        q, k, attributes<DotGeneralAttrs>{signature.score_dims, precision});
    const auto original_numerator =
        dot_general(scores, v, attributes<DotGeneralAttrs>{batch, precision});
    const auto original_denominator =
        reduce(scores, attributes<ReduceAttrs>{ReduceKind::Sum, Axes{2}});
    const auto summary_numerator = dot_general(
        q,
        dot_general(
            k, v,
            attributes<DotGeneralAttrs>{signature.summary_dims, precision}),
        attributes<DotGeneralAttrs>{batch, precision});
    const auto summary_denominator =
        dot_general(q,
                    reduce(k, attributes<ReduceAttrs>{ReduceKind::Sum,
                                                      signature.key_sum_axes}),
                    attributes<DotGeneralAttrs>{batch, precision});
    const std::string name =
        std::string("kernel-attention-summaries") + signature.suffix;
    rules.push_back(rule(name,
                         divide(original_numerator, D0(original_denominator)),
                         divide(summary_numerator, D0(summary_denominator)))
                        .when(kernel_guards));
    rules.push_back(
        rule(name + "-nested-broadcast",
             divide(original_numerator, D1(D0(original_denominator))),
             divide(summary_numerator, D1(D0(summary_denominator))))
            .when(kernel_guards));
  }

  // Move row-wise normalization after the dot. Explicitly match broadcasts;
  // arbitrary unary operator variables would not prove denominator invariance.
  const auto denominator = tensor_var("denominator");
  const auto division_dims = attribute_var<DotDimensions>("division_dims");
  const auto broadcast0 = attribute_var<BroadcastAttrs>("broadcast0"),
             broadcast1 = attribute_var<BroadcastAttrs>("broadcast1");
  const auto direct = broadcast_in_dim(denominator, broadcast0);
  for (bool nested : {false, true}) {
    const auto expanded =
        nested ? broadcast_in_dim(direct, broadcast1) : direct;
    rules.push_back(
        rule(nested ? "dot-divide-broadcast-nested" : "dot-divide-broadcast",
             dot_general(divide(x, expanded), v,
                         attributes<DotGeneralAttrs>{division_dims, precision}))
            .when(dot_division())
            .build([=](const Match& m, RhsBuilder& rhs) {
              auto axes = m[broadcast0].dimensions;
              if (nested) {
                const auto outer = m[broadcast1].dimensions;
                for (auto& axis : axes) axis = outer.at(axis);
              }
              return divideAfterDot(m, rhs, x, v, denominator, division_dims,
                                    precision, std::move(axes));
            }));
  }

  // ||A^T B||_F^2 = <AA^T, BB^T>. This isolated norm equation does not rewrite
  // Barlow's diagonal gather or eliminate the full loss's other consumers.
  const auto sample_gram_norm = reduce(
      multiply(
          dot_general(
              a, a, attributes<DotGeneralAttrs>{dot_dims({1}, {1}), precision}),
          dot_general(
              b, b,
              attributes<DotGeneralAttrs>{dot_dims({1}, {1}), precision})),
      attributes<ReduceAttrs>{ReduceKind::Sum, Axes{0, 1}});
  for (bool folded : {false, true}) {
    const auto correlation =
        folded
            ? dot_general(
                  a, b,
                  attributes<DotGeneralAttrs>{dot_dims({0}, {0}), precision})
            : dot_general(transpose(a, attributes<TransposeAttrs>{Axes{1, 0}}),
                          b, attributes<DotGeneralAttrs>{matrix, precision});
    rules.push_back(
        rule(folded ? "feature-gram-to-sample-gram-folded"
                    : "feature-gram-to-sample-gram",
             reduce(multiply(correlation, correlation),
                    attributes<ReduceAttrs>{ReduceKind::Sum, Axes{0, 1}}),
             sample_gram_norm)
            .when({rank(a, 2), rank(b, 2), dot_arithmetic()}));
  }

  // Sum((x - mean(x))^2) = sum(x^2) - sum(x)^2 / N. This deliberately
  // aggressive equation stays off even in the relaxed numerical preset.
  const auto count = tensor_var("count");
  const auto axes = attribute_var<Axes>("moment_axes");
  const auto mean_broadcast = attribute_var<BroadcastAttrs>("mean_broadcast"),
             count_broadcast = attribute_var<BroadcastAttrs>("count_broadcast"),
             kept_broadcast = attribute_var<BroadcastAttrs>("kept_broadcast");
  const auto sum_x = reduce(x, attributes<ReduceAttrs>{ReduceKind::Sum, axes});
  for (bool keepdims : {false, true}) {
    const auto mean =
        divide(keepdims ? broadcast_in_dim(sum_x, kept_broadcast) : sum_x,
               broadcast_in_dim(count, count_broadcast));
    const auto centered = subtract(x, broadcast_in_dim(mean, mean_broadcast));
    rules.push_back(
        rule(keepdims ? "centered-square-to-raw-moments-keepdims"
                      : "centered-square-to-raw-moments",
             reduce(multiply(centered, centered),
                    attributes<ReduceAttrs>{ReduceKind::Sum, axes}))
            .when({scalar(count), raw_moments()})
            .build([=](const Match& m, RhsBuilder& rhs) {
              auto xt = m.type(x);
              auto reduced = m[axes];
              if (!xt || !xt.hasStaticShape() || xt.getEncoding() ||
                  reduced.empty())
                return rhs.reject(
                    "raw moments require a static nonempty reduction");
              int64_t size = 1;
              for (auto axis : reduced) {
                auto length = xt.getDimSize(axis);
                if (length <= 0 ||
                    size > std::numeric_limits<int64_t>::max() / length)
                  return rhs.reject(
                      "raw moments require a positive bounded reduction size");
                size *= length;
              }
              auto literal = llvm::dyn_cast_or_null<mlir::DenseFPElementsAttr>(
                  m.constant(count));
              if (!literal || !literal.isSplat() ||
                  literal.getSplatValue<llvm::APFloat>().convertToDouble() !=
                      static_cast<double>(size))
                return rhs.reject("mean divisor must equal the reduction size");
              Axes surviving, sum_shape,
                  keep_shape(xt.getShape().begin(), xt.getShape().end());
              for (int64_t i = 0; i < xt.getRank(); ++i) {
                if (std::find(reduced.begin(), reduced.end(), i) ==
                    reduced.end()) {
                  surviving.push_back(i);
                  sum_shape.push_back(xt.getDimSize(i));
                } else {
                  keep_shape[i] = 1;
                }
              }
              const auto st =
                  mlir::RankedTensorType::get(sum_shape, xt.getElementType());
              auto nb = m[count_broadcast], mb = m[mean_broadcast];
              if (!nb.dimensions.empty() || mb.result_type != xt)
                return rhs.reject(
                    "raw moments require canonical mean broadcasts");
              if (keepdims) {
                const auto kt = mlir::RankedTensorType::get(
                    keep_shape, xt.getElementType());
                Axes identity(xt.getRank());
                std::iota(identity.begin(), identity.end(), 0);
                auto kb = m[kept_broadcast];
                if (kb != BroadcastAttrs{surviving, kt} ||
                    nb.result_type != kt || mb.dimensions != identity)
                  return rhs.reject(
                      "raw moments require canonical keepdims broadcasts");
              } else if (nb.result_type != st || mb.dimensions != surviving) {
                return rhs.reject(
                    "raw moments require canonical reduction broadcasts");
              }
              const auto value = rhs.ref(m[x]);
              const auto sum = rhs.make(
                  reduce, ReduceAttrs{ReduceKind::Sum, reduced}, value);
              const auto square = rhs.make(multiply, value, value);
              const auto sum_square = rhs.make(multiply, sum, sum);
              const auto divisor = rhs.make(
                  broadcast_in_dim, BroadcastAttrs{{}, st}, rhs.ref(m[count]));
              return rhs.make(
                  subtract,
                  rhs.make(reduce, ReduceAttrs{ReduceKind::Sum, reduced},
                           square),
                  rhs.make(divide, sum_square, divisor));
            }));
  }
  return compileRules(std::move(rules), policy, std::move(options));
}
}  // namespace joint_shard

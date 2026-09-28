#ifndef RESEARCH_JOINT_SHARD_BRIDGE_TENSORLANG_H_
#define RESEARCH_JOINT_SHARD_BRIDGE_TENSORLANG_H_
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <variant>
#include <vector>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "eggc/expr.hpp"
#include "eggc/language.hpp"
#include "eggc/pattern.hpp"

namespace joint_shard {

enum class OpKind {
  Input,
  Constant,
  Add,
  Subtract,
  Multiply,
  Divide,
  Maximum,
  Minimum,
  Negate,
  Exp,
  Log,
  Sqrt,
  Tanh,
  DotGeneral,
  Reduce,
  BroadcastInDim,
  Reshape,
  Transpose,
};

const char* stableHloName(OpKind kind);

struct NoAttrs {
  bool operator==(const NoAttrs&) const noexcept { return true; }
};
struct InputAttrs {
  unsigned index;
  mlir::RankedTensorType type;
  bool operator==(const InputAttrs& rhs) const {
    return index == rhs.index && type == rhs.type;
  }
};
struct DotGeneralAttrs {
  std::vector<int64_t> lhs_contracting;
  std::vector<int64_t> rhs_contracting;
  std::vector<int64_t> lhs_batching;
  std::vector<int64_t> rhs_batching;
  mlir::ArrayAttr precision_config;
  mlir::Attribute algorithm;
  // Explicit result type conservatively preserves dtype/encoding and dynamism.
  mlir::RankedTensorType result_type;
  // Manual producers may preserve extra attributes. The importer treats
  // unknown metadata as a boundary, and algebraic rules reject this dictionary
  // when nonempty. Managed dimensions/precision/algorithm cannot occur here.
  mlir::DictionaryAttr extra_attributes;
  bool operator==(const DotGeneralAttrs& rhs) const {
    return std::tie(lhs_contracting, rhs_contracting, lhs_batching,
                    rhs_batching, precision_config, algorithm, result_type,
                    extra_attributes) ==
           std::tie(rhs.lhs_contracting, rhs.rhs_contracting, rhs.lhs_batching,
                    rhs.rhs_batching, rhs.precision_config, rhs.algorithm,
                    rhs.result_type, rhs.extra_attributes);
  }
};
struct TransposeAttrs {
  std::vector<int64_t> permutation;
  bool operator==(const TransposeAttrs& rhs) const {
    return permutation == rhs.permutation;
  }
};
struct ReshapeAttrs {
  mlir::RankedTensorType result_type;
  bool operator==(const ReshapeAttrs& rhs) const {
    return result_type == rhs.result_type;
  }
};
struct BroadcastAttrs {
  std::vector<int64_t> dimensions;
  mlir::RankedTensorType result_type;
  bool operator==(const BroadcastAttrs& rhs) const {
    return dimensions == rhs.dimensions && result_type == rhs.result_type;
  }
};
struct ConstantAttrs {
  mlir::ElementsAttr value;
  bool operator==(const ConstantAttrs& rhs) const { return value == rhs.value; }
};
enum class ReduceKind { Sum, Max, Min, Product };
struct ReduceAttrs {
  ReduceKind kind;
  std::vector<int64_t> axes;
  // Only validated canonical identity-initialized reducers use this form.
  mlir::ElementsAttr initializer;
  bool operator==(const ReduceAttrs& rhs) const {
    return kind == rhs.kind && axes == rhs.axes &&
           initializer == rhs.initializer;
  }
};
using OpAttrs =
    std::variant<NoAttrs, InputAttrs, DotGeneralAttrs, TransposeAttrs,
                 ReshapeAttrs, BroadcastAttrs, ConstantAttrs, ReduceAttrs>;

// MLIR handles in attributes are context-owned. Keep the context alive through
// graph use/export. No descriptor table or encoded operator strings are needed.
struct TensorNode {
  // matches() compares the complete non-child identity, so egg-c may use its
  // memo table when every variable in a subpattern is already bound.
  static constexpr bool exact_matches = true;
  OpKind op;
  OpAttrs attrs;
  std::vector<eggc::Id> operands;
  using Discriminant = OpKind;
  OpKind discriminant() const { return op; }
  const std::vector<eggc::Id>& children() const { return operands; }
  std::vector<eggc::Id>& children_mut() { return operands; }
  bool matches(const TensorNode& other) const;
  bool operator==(const TensorNode& other) const;
  std::size_t hash() const;
  std::string format() const;
};
// One schema is shared by patterns, importer, analysis, and exporter.
struct OpSchema {
  OpKind op;
  std::string_view name;
  unsigned arity;
  bool attribute_free;
};
std::span<const OpSchema> opSchemas();
const OpSchema* opSchema(OpKind op);
const OpSchema* lookupOpSchema(std::string_view name);
bool validNodeSchema(const TensorNode& node);

using TensorRecExpr = eggc::RecExpr<TensorNode>;
using TensorPattern = eggc::Pattern<TensorNode>;

}  // namespace joint_shard

#endif  // RESEARCH_JOINT_SHARD_BRIDGE_TENSORLANG_H_

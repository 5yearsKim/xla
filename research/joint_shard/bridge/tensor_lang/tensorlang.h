#ifndef RESEARCH_JOINT_SHARD_BRIDGE_TENSORLANG_H_
#define RESEARCH_JOINT_SHARD_BRIDGE_TENSORLANG_H_
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "eggc/expr.hpp"
#include "eggc/language.hpp"

namespace joint_shard {
using Axes = std::vector<int64_t>;
enum class ReduceKind { Sum, Max, Min, Product };
enum class FieldPolicy { Required, Default, Inferred };

#define TENSOR_FIELD(record, type, name, policy) type name{};
#define TENSOR_VALUE(record, fields)                                    \
  struct record {                                                       \
    fields(TENSOR_FIELD, record) bool operator==(const record&) const = \
        default;                                                        \
  };
#define TENSOR_ATTRS TENSOR_VALUE
#define TENSOR_OP(kind, name, external, arity, attrs)
#include "research/joint_shard/bridge/tensor_lang/tensorlang.def"
#undef TENSOR_OP
#undef TENSOR_ATTRS
#undef TENSOR_VALUE
#undef TENSOR_FIELD

inline DotDimensions dot_dims(Axes lhs_contracting, Axes rhs_contracting,
                              Axes lhs_batching = {}, Axes rhs_batching = {}) {
  return {std::move(lhs_contracting), std::move(rhs_contracting),
          std::move(lhs_batching), std::move(rhs_batching)};
}

// Field descriptors refer to the actual record members. All users traverse
// these descriptors rather than maintaining their own list of attributes.
template <class Record>
struct AttributeSchema;
template <auto Member>
struct AttributeField {
  std::string_view name;
  FieldPolicy policy;
  static constexpr auto member = Member;
};
#define TENSOR_FIELD(record, type, name, policy) \
  AttributeField<&record::name>{#name, FieldPolicy::policy},
#define TENSOR_VALUE(record, fields)                                          \
  template <>                                                                 \
  struct AttributeSchema<record> {                                            \
    static constexpr auto members = std::tuple{fields(TENSOR_FIELD, record)}; \
  };
#define TENSOR_ATTRS TENSOR_VALUE
#define TENSOR_OP(kind, name, external, arity, attrs)
#include "research/joint_shard/bridge/tensor_lang/tensorlang.def"
#undef TENSOR_OP
#undef TENSOR_ATTRS
#undef TENSOR_VALUE
#undef TENSOR_FIELD

template <class Record, class Function>
void visitAttributeFields(Record&& record, Function&& function) {
  std::apply(
      [&](auto... field) {
        (function(field, record.*(decltype(field)::member)), ...);
      },
      AttributeSchema<std::remove_cvref_t<Record>>::members);
}

template <class Tuple>
struct AttributeVariant;
template <class... T>
struct AttributeVariant<std::tuple<T...>> {
  using type = std::variant<T...>;
};
// clang-format off
using OpAttributeRecords = decltype(std::tuple_cat(std::tuple<>{}
#define TENSOR_VALUE(record, fields)
#define TENSOR_ATTRS(record, fields) , std::tuple<record>{}
#define TENSOR_OP(kind, name, external, arity, attrs)
#include "research/joint_shard/bridge/tensor_lang/tensorlang.def"
#undef TENSOR_OP
#undef TENSOR_ATTRS
#undef TENSOR_VALUE
));
// clang-format on
using OpAttrs = AttributeVariant<OpAttributeRecords>::type;

enum class OpKind {
#define TENSOR_VALUE(record, fields)
#define TENSOR_ATTRS(record, fields)
#define TENSOR_OP(kind, name, external, arity, attrs) kind,
#include "research/joint_shard/bridge/tensor_lang/tensorlang.def"
#undef TENSOR_OP
#undef TENSOR_ATTRS
#undef TENSOR_VALUE
};
template <OpKind Kind>
struct OpTraits;
#define TENSOR_VALUE(record, fields)
#define TENSOR_ATTRS(record, fields)
#define TENSOR_OP(kind, callable, external, operands, record) \
  template <>                                                 \
  struct OpTraits<OpKind::kind> {                             \
    using Attributes = record;                                \
    static constexpr unsigned arity = operands;               \
    static constexpr std::string_view name = external;        \
  };
#include "research/joint_shard/bridge/tensor_lang/tensorlang.def"
#undef TENSOR_OP
#undef TENSOR_ATTRS
#undef TENSOR_VALUE

const char* stableHloName(OpKind kind);

// MLIR handles in attributes are context-owned. Keep the context alive through
// graph use/export. Nodes store native handles, not encoded operator strings.
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
// Runtime views are generated from the same definition as compile-time traits.
struct OpSchema {
  OpKind op;
  std::string_view name;
  unsigned arity;
  bool attribute_free;
  bool (*accepts_attributes)(const OpAttrs&);
};
std::span<const OpSchema> opSchemas();
const OpSchema* opSchema(OpKind op);
const OpSchema* lookupOpSchema(std::string_view name);
bool validNodeSchema(const TensorNode& node);

using TensorRecExpr = eggc::RecExpr<TensorNode>;
}  // namespace joint_shard
#endif

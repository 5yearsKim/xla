#ifndef RESEARCH_JOINT_SHARD_BRIDGE_TENSOR_LANG_TENSOR_PATTERNS_H_
#define RESEARCH_JOINT_SHARD_BRIDGE_TENSOR_LANG_TENSOR_PATTERNS_H_

#include <any>
#include <concepts>
#include <functional>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <utility>
#include <variant>
#include <vector>

#include "research/joint_shard/bridge/tensor_lang/tensor_attributes.h"
#include "research/joint_shard/bridge/tensor_lang/tensor_rewrites.h"

namespace joint_shard::semantic_detail {
struct Term;
struct Predicate;
struct Rule;
struct Bindings;
struct Prepared;
struct PatternAccess;
}  // namespace joint_shard::semantic_detail

namespace joint_shard::patterns {
class Pattern {
 public:
  // Patterns are immutable and can be shared by multiple rule definitions.
  Pattern(const Pattern&) = default;
  Pattern(Pattern&&) = default;
  Pattern& operator=(const Pattern&) = default;
  Pattern& operator=(Pattern&&) = default;

 private:
  explicit Pattern(std::shared_ptr<const semantic_detail::Term> term)
      : term_(std::move(term)) {}
  std::shared_ptr<const semantic_detail::Term> term_;
  friend struct semantic_detail::PatternAccess;
};
struct TensorVar {
  std::string name;
  operator Pattern() const;
};
struct OperatorVar {
  std::string name;
  unsigned arity;
  template <class... Args>
  Pattern operator()(Args&&... args) const {
    return call({Pattern(std::forward<Args>(args))...});
  }

 private:
  Pattern call(std::vector<Pattern> operands) const;
};
inline TensorVar tensor_var(std::string name) { return {std::move(name)}; }
inline OperatorVar operator_var(std::string name, unsigned arity) {
  return {std::move(name), arity};
}

template <class T>
struct AttributeVar {
  std::string name;
};
template <class T>
AttributeVar<T> attribute_var(std::string name) {
  return {std::move(name)};
}
// Attribute storage is type-erased, so neither the matcher nor callbacks need
// an independently maintained variant of TensorLang's attribute types.
class AttributeValue {
 public:
  template <class T>
  explicit AttributeValue(T value)
      : storage_(std::move(value)),
        type_(typeid(T)),
        equal_([](const std::any& a, const std::any& b) {
          return std::any_cast<const T&>(a) == std::any_cast<const T&>(b);
        }),
        format_([](const std::any& a) {
          return formatAttributeValue(std::any_cast<const T&>(a));
        }) {}
  template <class T>
  const T* get() const {
    return std::any_cast<T>(&storage_);
  }
  bool operator==(const AttributeValue& other) const {
    return type_ == other.type_ && equal_(storage_, other.storage_);
  }
  std::string format() const {
    return std::string(type_.name()) + ':' + format_(storage_);
  }

 private:
  std::any storage_;
  std::type_index type_;
  bool (*equal_)(const std::any&, const std::any&);
  std::string (*format_)(const std::any&);
};
struct Wildcard {};
struct Inferred {};
struct Missing {};
inline constexpr Wildcard any_attribute{};
using AttributeExpression =
    std::variant<AttributeValue, std::string, Wildcard, Inferred, Missing>;
template <class T>
class Attribute {
 public:
  Attribute(T value) : expression_(AttributeValue(std::move(value))) {}
  Attribute(AttributeVar<T> variable) : expression_(std::move(variable.name)) {}
  Attribute(Wildcard value) : expression_(value) {}
  static Attribute defaultFor(FieldPolicy policy) {
    if (policy == FieldPolicy::Default) return Attribute(T{});
    return Attribute(policy == FieldPolicy::Inferred
                         ? AttributeExpression(Inferred{})
                         : AttributeExpression(Missing{}));
  }
  const AttributeExpression& expression() const { return expression_; }

 private:
  explicit Attribute(AttributeExpression value)
      : expression_(std::move(value)) {}
  AttributeExpression expression_;
};

// Each field specification is generated from the same declaration as the
// concrete attribute record. Callers can set literals, captures or wildcards.
template <class Record>
struct attributes;
#define TENSOR_FIELD(record, type, name, policy) \
  Attribute<type> name = Attribute<type>::defaultFor(FieldPolicy::policy);
#define TENSOR_VALUE(record, fields)
#define TENSOR_ATTRS(record, fields) \
  template <>                        \
  struct attributes<record> {        \
    fields(TENSOR_FIELD, record)     \
  };
#define TENSOR_OP(kind, name, external, arity, attrs)
#include "research/joint_shard/bridge/tensor_lang/tensorlang.def"
#undef TENSOR_OP
#undef TENSOR_ATTRS
#undef TENSOR_VALUE
#undef TENSOR_FIELD

#define TENSOR_FIELD(record, type, name, policy)                      \
  function(AttributeField<&record::name>{#name, FieldPolicy::policy}, \
           attrs.name);
#define TENSOR_VALUE(record, fields)
#define TENSOR_ATTRS(record, fields)                       \
  template <class Function>                                \
  void visitPatternFields(const attributes<record>& attrs, \
                          Function&& function) {           \
    fields(TENSOR_FIELD, record)                           \
  }
#define TENSOR_OP(kind, name, external, arity, attrs)
#include "research/joint_shard/bridge/tensor_lang/tensorlang.def"
#undef TENSOR_OP
#undef TENSOR_ATTRS
#undef TENSOR_VALUE
#undef TENSOR_FIELD

struct AttributeConstraint {
  std::string_view field;
  std::type_index type;
  AttributeExpression expression;
  AttributeValue (*read)(const OpAttrs&);
  void (*write)(OpAttrs&, const AttributeValue&);
};
template <class Record>
class AttributeSpec {
 public:
  AttributeSpec(attributes<Record> fields = {}) {
    visitPatternFields(fields, [&](auto field, const auto& value) {
      using T = std::remove_cvref_t<decltype(std::declval<Record>().*
                                             (decltype(field)::member))>;
      constraints_.push_back(
          {field.name, typeid(T), value.expression(),
           [](const OpAttrs& attrs) {
             return AttributeValue(std::get<Record>(attrs).*
                                   (decltype(field)::member));
           },
           [](OpAttrs& attrs, const AttributeValue& value) {
             std::get<Record>(attrs).*(decltype(field)::member) =
                 *value.template get<T>();
           }});
    });
  }
  AttributeSpec(Record value) { whole(Attribute<Record>(std::move(value))); }
  AttributeSpec(AttributeVar<Record> variable) {
    whole(Attribute<Record>(std::move(variable)));
  }
  const std::vector<AttributeConstraint>& constraints() const {
    return constraints_;
  }

 private:
  void whole(const Attribute<Record>& value) {
    constraints_.push_back({"*", typeid(Record), value.expression(),
                            [](const OpAttrs& attrs) {
                              return AttributeValue(std::get<Record>(attrs));
                            },
                            [](OpAttrs& attrs, const AttributeValue& value) {
                              attrs = *value.get<Record>();
                            }});
  }
  std::vector<AttributeConstraint> constraints_;
};

namespace detail {
Pattern concrete(OpKind op, OpAttrs initial_attributes,
                 std::vector<Pattern> operands,
                 std::vector<AttributeConstraint> attributes);
}
// Generic adapters add pattern calls to TensorLang's structural traits. Arity
// and attribute types are supplied entirely by the language definition.
template <OpKind Kind>
struct Operator {
  using Traits = OpTraits<Kind>;
  using Attrs = typename Traits::Attributes;
  static constexpr auto kind = Kind;
  template <class... Args>
  static consteval bool accepts() {
    using Tuple = std::tuple<Args...>;
    if constexpr (sizeof...(Args) == Traits::arity ||
                  sizeof...(Args) == Traits::arity + 1) {
      constexpr bool operands = []<std::size_t... I>(
                                    std::index_sequence<I...>) {
        return (std::convertible_to<std::tuple_element_t<I, Tuple>, Pattern> &&
                ...);
      }(std::make_index_sequence<Traits::arity>{});
      if constexpr (sizeof...(Args) == Traits::arity)
        return operands;
      else
        return operands &&
               std::convertible_to<std::tuple_element_t<Traits::arity, Tuple>,
                                   AttributeSpec<Attrs>>;
    } else
      return false;
  }
  template <class... Args>
    requires(accepts<Args...>())
  Pattern operator()(Args&&... args) const {
    auto tuple = std::forward_as_tuple(std::forward<Args>(args)...);
    if constexpr (sizeof...(Args) == Traits::arity)
      return call(tuple, AttributeSpec<Attrs>{},
                  std::make_index_sequence<Traits::arity>{});
    else
      return call(tuple, AttributeSpec<Attrs>(std::get<Traits::arity>(tuple)),
                  std::make_index_sequence<Traits::arity>{});
  }

 private:
  template <class Tuple, std::size_t... I>
  Pattern call(Tuple& args, const AttributeSpec<Attrs>& attrs,
               std::index_sequence<I...>) const {
    return detail::concrete(Kind, Attrs{}, {Pattern(std::get<I>(args))...},
                            attrs.constraints());
  }
};
Pattern scale(Pattern scalar, Pattern tensor);

class Guard {
 private:
  explicit Guard(std::shared_ptr<const semantic_detail::Predicate> predicate)
      : predicate_(std::move(predicate)) {}
  std::shared_ptr<const semantic_detail::Predicate> predicate_;
  friend struct semantic_detail::PatternAccess;
};
Guard property(OperatorVar op, OpProperty property);
Guard elementwise(OperatorVar op);
Guard commutative(OperatorVar op);
Guard associative(OperatorVar op);
Guard involution(OperatorVar op);
Guard linear_in(OperatorVar op, unsigned operand);
Guard homogeneous_in(OperatorVar op, unsigned operand);
Guard scalar(TensorVar tensor);
Guard uniform(TensorVar tensor);
Guard rank(TensorVar tensor, unsigned rank);
Guard dot_reassociation();
Guard sum_dot_interchange();
// Numerical contract for explicit equations involving dot arithmetic.
Guard dot_arithmetic();
// Floating division movement additionally assumes nonzero finite denominators
// and no intermediate overflow/underflow that invalidates the equation.
Guard dot_division();
Guard raw_moments();

class Match {
 public:
  eggc::Id operator[](const TensorVar& variable) const;
  mlir::RankedTensorType type(const TensorVar& variable) const;
  mlir::ElementsAttr constant(const TensorVar& variable) const;
  template <class T>
  T operator[](const AttributeVar<T>& variable) const {
    auto value = attribute(variable.name);
    if (const auto* typed = value.template get<T>()) return *typed;
    throw std::invalid_argument("attribute variable type mismatch: " +
                                variable.name);
  }

 private:
  using Value = AttributeValue;
  Match(const semantic_detail::Bindings& bindings, const TensorEGraph& graph)
      : bindings_(bindings), graph_(graph) {}
  Value attribute(const std::string& name) const;
  const semantic_detail::Bindings& bindings_;
  const TensorEGraph& graph_;
  friend struct semantic_detail::PatternAccess;
};

class Expression {
 private:
  std::shared_ptr<const semantic_detail::Prepared> prepared_;
  std::string error_;
  friend class RhsBuilder;
  friend struct semantic_detail::PatternAccess;
};
// A callback builds a temporary expression. No operation below mutates the
// e-graph; the rule compiler validates the complete result before insertion.
class RhsBuilder {
 public:
  Expression ref(eggc::Id id) const;
  Expression reject(std::string reason) const;
  template <OpKind Kind, class... Args>
    requires(sizeof...(Args) == OpTraits<Kind>::arity &&
             (std::same_as<std::remove_cvref_t<Args>, Expression> && ...))
  Expression make(Operator<Kind>, typename OpTraits<Kind>::Attributes attrs,
                  Args&&... operands) const {
    return operation(Kind, std::move(attrs), {std::forward<Args>(operands)...});
  }
  template <OpKind Kind, class... Args>
    requires(std::same_as<typename OpTraits<Kind>::Attributes, NoAttrs> &&
             sizeof...(Args) == OpTraits<Kind>::arity &&
             (std::same_as<std::remove_cvref_t<Args>, Expression> && ...))
  Expression make(Operator<Kind> op, Args&&... operands) const {
    return make(op, NoAttrs{}, std::forward<Args>(operands)...);
  }
  Expression apply(const OperatorVar& op,
                   std::vector<Expression> operands) const;

 private:
  Expression operation(OpKind op, OpAttrs attrs,
                       std::vector<Expression> operands) const;
  RhsBuilder(const semantic_detail::Bindings& bindings,
             const TensorEGraph& graph)
      : bindings_(bindings), graph_(graph) {}
  const semantic_detail::Bindings& bindings_;
  const TensorEGraph& graph_;
  friend struct semantic_detail::PatternAccess;
};
using RhsCallback = std::function<Expression(const Match&, RhsBuilder&)>;

class RuleDefinition {
 public:
  RuleDefinition& when(Guard guard);
  RuleDefinition& when(std::initializer_list<Guard> guards);
  RuleDefinition& build(RhsCallback callback);

 private:
  explicit RuleDefinition(std::shared_ptr<const semantic_detail::Rule> rule)
      : rule_(std::move(rule)) {}
  std::shared_ptr<const semantic_detail::Rule> rule_;
  friend struct semantic_detail::PatternAccess;
};
RuleDefinition rule(std::string name, Pattern lhs, Pattern rhs);
RuleDefinition rule(std::string name, Pattern lhs);
}  // namespace joint_shard::patterns

namespace joint_shard::tensorlang::ops {
#define TENSOR_VALUE(record, fields)
#define TENSOR_ATTRS(record, fields)
#define TENSOR_OP(kind, name, external, arity, attrs) \
  inline constexpr patterns::Operator<OpKind::kind> name{};
#include "research/joint_shard/bridge/tensor_lang/tensorlang.def"
#undef TENSOR_OP
#undef TENSOR_ATTRS
#undef TENSOR_VALUE
}  // namespace joint_shard::tensorlang::ops
#endif

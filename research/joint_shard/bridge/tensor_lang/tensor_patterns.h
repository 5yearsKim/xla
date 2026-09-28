#ifndef RESEARCH_JOINT_SHARD_BRIDGE_TENSOR_LANG_TENSOR_PATTERNS_H_
#define RESEARCH_JOINT_SHARD_BRIDGE_TENSOR_LANG_TENSOR_PATTERNS_H_

#include <functional>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

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
using Axes = std::vector<int64_t>;
struct DotDimensions {
  Axes lhs_contracting;
  Axes rhs_contracting;
  Axes lhs_batching;
  Axes rhs_batching;
  bool operator==(const DotDimensions&) const = default;
};
inline DotDimensions dot_dims(Axes lhs_contracting, Axes rhs_contracting,
                              Axes lhs_batching = {}, Axes rhs_batching = {}) {
  return {std::move(lhs_contracting), std::move(rhs_contracting),
          std::move(lhs_batching), std::move(rhs_batching)};
}

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
template <class T>
class Attribute {
 public:
  Attribute(T value = {}) : value_(std::move(value)) {}
  Attribute(AttributeVar<T> variable) : value_(std::move(variable)) {}
  const std::variant<T, AttributeVar<T>>& value() const { return value_; }

 private:
  std::variant<T, AttributeVar<T>> value_;
};

// The low-level constructor is for attribute-free concrete operators. Arity is
// checked against TensorLang's schema. Attribute-bearing operations are typed.
Pattern operation(OpKind op, std::vector<Pattern> operands);
Pattern add(Pattern a, Pattern b);
Pattern subtract(Pattern a, Pattern b);
Pattern multiply(Pattern a, Pattern b);
Pattern divide(Pattern a, Pattern b);
Pattern maximum(Pattern a, Pattern b);
Pattern minimum(Pattern a, Pattern b);
Pattern negate(Pattern x);
Pattern exp(Pattern x);
Pattern log(Pattern x);
Pattern sqrt(Pattern x);
Pattern tanh(Pattern x);
Pattern scale(Pattern scalar, Pattern tensor);
Pattern dot_general(Pattern a, Pattern b, Attribute<DotDimensions> dimensions,
                    Attribute<mlir::ArrayAttr> precision = {});
Pattern reduce(Pattern x, Attribute<ReduceKind> kind, Attribute<Axes> axes);
Pattern transpose(Pattern x, Attribute<Axes> permutation);
// Capture both the dimension mapping and destination type. Moving a broadcast
// to another tensor shape requires constructing new attributes in a callback.
Pattern broadcast_in_dim(Pattern x, Attribute<BroadcastAttrs> attrs);

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
    if (const auto* typed = std::get_if<T>(&value)) return *typed;
    throw std::invalid_argument("attribute variable type mismatch: " +
                                variable.name);
  }

 private:
  using Value = std::variant<Axes, ReduceKind, mlir::ArrayAttr, DotDimensions,
                             BroadcastAttrs>;
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
  Expression operation(OpKind op, OpAttrs attrs,
                       std::vector<Expression> operands) const;
  Expression add(Expression a, Expression b) const;
  Expression negate(Expression x) const;
  Expression divide(Expression a, Expression b) const;
  Expression dot_general(Expression a, Expression b, DotDimensions dimensions,
                         mlir::ArrayAttr precision = {}) const;
  Expression reduce(Expression x, ReduceKind kind, Axes axes) const;
  Expression transpose(Expression x, Axes permutation) const;
  Expression broadcast_in_dim(Expression x, BroadcastAttrs attrs) const;
  Expression apply(const OperatorVar& op,
                   std::vector<Expression> operands) const;

 private:
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

#endif

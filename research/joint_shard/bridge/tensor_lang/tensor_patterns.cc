#include "research/joint_shard/bridge/tensor_lang/tensor_patterns.h"

#include <array>
#include <stdexcept>
#include <type_traits>

#include "research/joint_shard/bridge/tensor_lang/semantic_rule_internal.h"

namespace joint_shard::patterns {
namespace {
using semantic_detail::AttributeField;
using semantic_detail::AttributePattern;
using semantic_detail::AttributeValue;
using semantic_detail::PatternAccess;
using semantic_detail::PredicateKind;
using semantic_detail::Term;
using semantic_detail::TermKind;

template <class T>
AttributePattern field(AttributeField kind, const Attribute<T>& value) {
  AttributePattern result{kind, AttributeValue{}};
  std::visit(
      [&](const auto& v) {
        if constexpr (std::is_same_v<std::decay_t<decltype(v)>, T>)
          result.expression = AttributeValue(v);
        else
          result.expression = v.name;
      },
      value.value());
  return result;
}
Pattern concrete(OpKind op, std::vector<Pattern> operands,
                 std::vector<AttributePattern> attributes = {}) {
  auto* schema = opSchema(op);
  if (!schema || operands.size() != schema->arity)
    throw std::invalid_argument(
        "concrete pattern has invalid operator or arity");
  Term term;
  term.kind = TermKind::ConcreteOperator;
  term.op = op;
  term.attributes = std::move(attributes);
  for (const auto& operand : operands)
    term.children.push_back(PatternAccess::term(operand));
  return PatternAccess::pattern(std::move(term));
}
Guard valueGuard(PredicateKind kind, TensorVar tensor,
                 std::optional<unsigned> index = {}) {
  return PatternAccess::guard({kind, std::move(tensor.name), index});
}
}  // namespace

TensorVar::operator Pattern() const {
  Term term;
  term.name = name;
  term.kind = TermKind::TensorVariable;
  return PatternAccess::pattern(std::move(term));
}
Pattern OperatorVar::call(std::vector<Pattern> operands) const {
  if (operands.size() != arity)
    throw std::invalid_argument("wrong arity for operator variable " + name);
  Term term;
  term.name = name;
  term.kind = TermKind::OperatorVariable;
  for (const auto& operand : operands)
    term.children.push_back(PatternAccess::term(operand));
  return PatternAccess::pattern(std::move(term));
}
Pattern operation(OpKind op, std::vector<Pattern> operands) {
  auto* schema = opSchema(op);
  if (!schema || !schema->attribute_free)
    throw std::invalid_argument(
        "use a typed constructor for attribute-bearing patterns");
  return concrete(op, std::move(operands));
}
Pattern add(Pattern a, Pattern b) { return operation(OpKind::Add, {a, b}); }
Pattern subtract(Pattern a, Pattern b) {
  return operation(OpKind::Subtract, {a, b});
}
Pattern multiply(Pattern a, Pattern b) {
  return operation(OpKind::Multiply, {a, b});
}
Pattern divide(Pattern a, Pattern b) {
  return operation(OpKind::Divide, {a, b});
}
Pattern negate(Pattern x) { return operation(OpKind::Negate, {x}); }
Pattern scale(Pattern scalar, Pattern tensor) {
  Term term;
  term.kind = TermKind::Scale;
  term.children = {PatternAccess::term(scalar), PatternAccess::term(tensor)};
  return PatternAccess::pattern(std::move(term));
}
Pattern dot_general(Pattern a, Pattern b, Attribute<DotDimensions> dimensions,
                    Attribute<mlir::ArrayAttr> precision) {
  return concrete(OpKind::DotGeneral, {a, b},
                  {field(AttributeField::DotDimensions, dimensions),
                   field(AttributeField::PrecisionConfig, precision)});
}
Pattern reduce(Pattern x, Attribute<ReduceKind> kind, Attribute<Axes> axes) {
  return concrete(OpKind::Reduce, {x},
                  {field(AttributeField::ReduceKind, kind),
                   field(AttributeField::Axes, axes)});
}
Pattern transpose(Pattern x, Attribute<Axes> permutation) {
  return concrete(OpKind::Transpose, {x},
                  {field(AttributeField::Permutation, permutation)});
}
Pattern broadcast_in_dim(Pattern x, Attribute<BroadcastAttrs> attrs) {
  return concrete(OpKind::BroadcastInDim, {x},
                  {field(AttributeField::Broadcast, attrs)});
}
Guard property(OperatorVar op, OpProperty property) {
  PredicateKind kind = std::visit(
      [](const auto& p) -> PredicateKind {
        using T = std::decay_t<decltype(p)>;
        if constexpr (std::is_same_v<T, Elementwise>)
          return PredicateKind::Elementwise;
        else if constexpr (std::is_same_v<T, Commutative>)
          return PredicateKind::Commutative;
        else if constexpr (std::is_same_v<T, Associative>)
          return PredicateKind::Associative;
        else if constexpr (std::is_same_v<T, Involution>)
          return PredicateKind::Involution;
        else if constexpr (std::is_same_v<T, LinearIn>)
          return PredicateKind::LinearIn;
        else
          return PredicateKind::HomogeneousIn;
      },
      property);
  std::optional<unsigned> operand;
  if (auto* p = std::get_if<LinearIn>(&property)) operand = p->operand;
  if (auto* p = std::get_if<HomogeneousIn>(&property)) operand = p->operand;
  if (operand && *operand >= op.arity)
    throw std::invalid_argument("property operand exceeds operator arity: " +
                                op.name);
  return PatternAccess::guard({kind, std::move(op.name), operand});
}
Guard elementwise(OperatorVar op) { return property(op, Elementwise{}); }
Guard commutative(OperatorVar op) { return property(op, Commutative{}); }
Guard associative(OperatorVar op) { return property(op, Associative{}); }
Guard involution(OperatorVar op) { return property(op, Involution{}); }
Guard linear_in(OperatorVar op, unsigned operand) {
  return property(op, LinearIn{operand});
}
Guard homogeneous_in(OperatorVar op, unsigned operand) {
  return property(op, HomogeneousIn{operand});
}
Guard scalar(TensorVar tensor) {
  return valueGuard(PredicateKind::Scalar, tensor);
}
Guard uniform(TensorVar tensor) {
  return valueGuard(PredicateKind::Uniform, tensor);
}
Guard rank(TensorVar tensor, unsigned rank) {
  return valueGuard(PredicateKind::Rank, tensor, rank);
}
Guard dot_reassociation() {
  return PatternAccess::guard({PredicateKind::DotReassociation, {}, {}});
}
Guard sum_dot_interchange() {
  return PatternAccess::guard({PredicateKind::SumDotInterchange, {}, {}});
}

Guard dot_arithmetic() {
  return PatternAccess::guard({PredicateKind::DotArithmetic, {}, {}});
}
Guard dot_division() {
  return PatternAccess::guard({PredicateKind::DotDivision, {}, {}});
}
Guard raw_moments() {
  return PatternAccess::guard({PredicateKind::RawMoments, {}, {}});
}

RuleDefinition rule(std::string name, Pattern lhs, Pattern rhs) {
  return PatternAccess::definition({std::move(name),
                                    PatternAccess::term(lhs),
                                    PatternAccess::term(rhs),
                                    {},
                                    {}});
}
RuleDefinition rule(std::string name, Pattern lhs) {
  return PatternAccess::definition(
      {std::move(name), PatternAccess::term(lhs), {}, {}, {}});
}
RuleDefinition& RuleDefinition::when(Guard guard) {
  auto copy = *rule_;
  copy.predicates.push_back(PatternAccess::predicate(guard));
  rule_ = std::make_shared<const semantic_detail::Rule>(std::move(copy));
  return *this;
}
RuleDefinition& RuleDefinition::when(std::initializer_list<Guard> guards) {
  for (const auto& guard : guards) when(guard);
  return *this;
}
RuleDefinition& RuleDefinition::build(RhsCallback callback) {
  if (rule_->rhs || rule_->builder || !callback)
    throw std::invalid_argument("rule must have exactly one RHS definition: " +
                                rule_->name);
  auto copy = *rule_;
  copy.builder = std::move(callback);
  rule_ = std::make_shared<const semantic_detail::Rule>(std::move(copy));
  return *this;
}

eggc::Id Match::operator[](const TensorVar& variable) const {
  auto found = bindings_.tensors.find(variable.name);
  if (found == bindings_.tensors.end())
    throw std::invalid_argument("unbound tensor variable " + variable.name);
  return graph_.find(found->second);
}
mlir::RankedTensorType Match::type(const TensorVar& variable) const {
  return tensorFacts(graph_, (*this)[variable]).type;
}
mlir::ElementsAttr Match::constant(const TensorVar& variable) const {
  return tensorFacts(graph_, (*this)[variable]).constant;
}
Match::Value Match::attribute(const std::string& name) const {
  auto found = bindings_.attributes.find(name);
  if (found == bindings_.attributes.end())
    throw std::invalid_argument("unbound attribute variable " + name);
  return found->second;
}
Expression RhsBuilder::reject(std::string reason) const {
  Expression result;
  result.error_ = std::move(reason);
  return result;
}
Expression RhsBuilder::ref(eggc::Id id) const {
  semantic_detail::Prepared result;
  result.reference = graph_.find(id);
  result.facts = tensorFacts(graph_, id);
  Expression expression;
  expression.prepared_ =
      std::make_shared<const semantic_detail::Prepared>(std::move(result));
  return expression;
}
Expression RhsBuilder::operation(OpKind op, OpAttrs attrs,
                                 std::vector<Expression> operands) const {
  semantic_detail::Prepared result;
  std::vector<TensorFacts> facts;
  for (const auto& operand : operands) {
    if (!operand.prepared_) return operand;
    result.children.push_back(*operand.prepared_);
    facts.push_back(operand.prepared_->facts);
  }
  result.node = {op, std::move(attrs),
                 std::vector<eggc::Id>(operands.size(), 0)};
  auto inference = inferTensorNode(result.node, facts);
  if (!inference.valid()) return reject("RHS: " + inference.reason);
  result.facts = inference.facts;
  Expression expression;
  expression.prepared_ =
      std::make_shared<const semantic_detail::Prepared>(std::move(result));
  return expression;
}
Expression RhsBuilder::negate(Expression x) const {
  return operation(OpKind::Negate, NoAttrs{}, {x});
}
Expression RhsBuilder::divide(Expression a, Expression b) const {
  return operation(OpKind::Divide, NoAttrs{}, {a, b});
}
Expression RhsBuilder::dot_general(Expression a, Expression b,
                                   DotDimensions dims,
                                   mlir::ArrayAttr precision) const {
  if (!a.prepared_) return a;
  if (!b.prepared_) return b;
  DotGeneralAttrs attrs{dims.lhs_contracting,
                        dims.rhs_contracting,
                        dims.lhs_batching,
                        dims.rhs_batching,
                        precision,
                        {},
                        {},
                        {}};
  std::array<TensorFacts, 2> operands{a.prepared_->facts, b.prepared_->facts};
  auto inferred = inferDotResultType(attrs, operands);
  if (!inferred.valid()) return reject("RHS dot_general: " + inferred.reason);
  attrs.result_type = inferred.facts.type;
  return operation(OpKind::DotGeneral, std::move(attrs), {a, b});
}
Expression RhsBuilder::reduce(Expression x, ReduceKind kind, Axes axes) const {
  if (!x.prepared_) return x;
  auto type = x.prepared_->facts.type;
  if (!type) return reject("RHS reduce requires a known type");
  auto initializer = canonicalReductionInitializer(kind, type.getElementType());
  return operation(OpKind::Reduce,
                   ReduceAttrs{kind, std::move(axes), initializer}, {x});
}
Expression RhsBuilder::transpose(Expression x, Axes permutation) const {
  return operation(OpKind::Transpose, TransposeAttrs{std::move(permutation)},
                   {x});
}
Expression RhsBuilder::broadcast_in_dim(Expression x,
                                        BroadcastAttrs attrs) const {
  return operation(OpKind::BroadcastInDim, std::move(attrs), {x});
}
Expression RhsBuilder::apply(const OperatorVar& op,
                             std::vector<Expression> operands) const {
  auto found = bindings_.operators.find(op.name);
  if (found == bindings_.operators.end())
    return reject("unbound operator variable " + op.name);
  if (operands.size() != op.arity)
    return reject("wrong RHS operator arity: " + op.name);
  const auto& node = found->second.front().node;
  auto expression = operation(node.op, node.attrs, std::move(operands));
  if (expression.prepared_) {
    auto prepared = *expression.prepared_;
    prepared.operator_variable = op.name;
    expression.prepared_ =
        std::make_shared<const semantic_detail::Prepared>(std::move(prepared));
  }
  return expression;
}
}  // namespace joint_shard::patterns

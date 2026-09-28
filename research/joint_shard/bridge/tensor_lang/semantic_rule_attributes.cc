#include <stdexcept>

#include "llvm/Support/raw_ostream.h"
#include "research/joint_shard/bridge/tensor_lang/semantic_rule_internal.h"

namespace joint_shard::semantic_detail {
namespace {
AttributeValue readAttribute(const TensorNode& node, AttributeField field) {
  if (auto* dot = std::get_if<DotGeneralAttrs>(&node.attrs)) {
    if (field == AttributeField::DotDimensions)
      return patterns::DotDimensions{dot->lhs_contracting, dot->rhs_contracting,
                                     dot->lhs_batching, dot->rhs_batching};
    if (field == AttributeField::PrecisionConfig) return dot->precision_config;
  }
  if (auto* reduce = std::get_if<ReduceAttrs>(&node.attrs)) {
    if (field == AttributeField::ReduceKind) return reduce->kind;
    if (field == AttributeField::Axes) return reduce->axes;
  }
  if (auto* transpose = std::get_if<TransposeAttrs>(&node.attrs))
    if (field == AttributeField::Permutation) return transpose->permutation;
  if (auto* broadcast = std::get_if<BroadcastAttrs>(&node.attrs))
    if (field == AttributeField::Broadcast) return *broadcast;
  throw std::logic_error("invalid typed attribute field");
}
AttributeValue resolve(const AttributePattern& pattern,
                       const Bindings& bindings) {
  if (auto* variable = std::get_if<std::string>(&pattern.expression))
    return bindings.attributes.at(*variable);
  return std::get<AttributeValue>(pattern.expression);
}
}  // namespace

bool matchAttributes(const Term& term, const TensorNode& node,
                     Bindings& bindings) {
  if (auto* dot = std::get_if<DotGeneralAttrs>(&node.attrs))
    if (dot->algorithm ||
        (dot->extra_attributes && !dot->extra_attributes.empty()))
      return false;
  for (const auto& pattern : term.attributes) {
    auto value = readAttribute(node, pattern.field);
    if (auto* variable = std::get_if<std::string>(&pattern.expression)) {
      auto [it, inserted] = bindings.attributes.emplace(*variable, value);
      if (!inserted && it->second != value) return false;
    } else if (std::get<AttributeValue>(pattern.expression) != value)
      return false;
  }
  return true;
}
std::optional<TensorNode> buildConcreteNode(
    const Term& term, const Bindings& bindings,
    std::span<const TensorFacts> operands, std::string& rejection) {
  TensorNode node{term.op, NoAttrs{},
                  std::vector<eggc::Id>(operands.size(), 0)};
  if (opSchema(term.op)->attribute_free) return node;
  const auto value = [&](AttributeField name) -> AttributeValue {
    for (const auto& field : term.attributes)
      if (field.field == name) return resolve(field, bindings);
    throw std::logic_error("missing typed semantic attribute");
  };
  if (term.op == OpKind::DotGeneral) {
    auto dims =
        std::get<patterns::DotDimensions>(value(AttributeField::DotDimensions));
    DotGeneralAttrs attrs{
        dims.lhs_contracting,
        dims.rhs_contracting,
        dims.lhs_batching,
        dims.rhs_batching,
        std::get<mlir::ArrayAttr>(value(AttributeField::PrecisionConfig)),
        {},
        {},
        {}};
    auto inferred = inferDotResultType(attrs, operands);
    if (!inferred.valid()) {
      rejection = "RHS dot_general: " + inferred.reason;
      return std::nullopt;
    }
    attrs.result_type = inferred.facts.type;
    node.attrs = std::move(attrs);
  } else if (term.op == OpKind::Reduce) {
    if (operands.size() != 1 || !operands[0].type) {
      rejection = "RHS reduce requires a known operand type";
      return std::nullopt;
    }
    auto kind = std::get<ReduceKind>(value(AttributeField::ReduceKind));
    auto initializer =
        canonicalReductionInitializer(kind, operands[0].type.getElementType());
    if (!initializer) {
      rejection = "RHS reduce has no supported canonical identity";
      return std::nullopt;
    }
    node.attrs =
        ReduceAttrs{kind, std::get<patterns::Axes>(value(AttributeField::Axes)),
                    initializer};
  } else if (term.op == OpKind::Transpose) {
    node.attrs = TransposeAttrs{
        std::get<patterns::Axes>(value(AttributeField::Permutation))};
  } else if (term.op == OpKind::BroadcastInDim) {
    node.attrs = std::get<BroadcastAttrs>(value(AttributeField::Broadcast));
  } else
    throw std::logic_error("unsupported typed concrete operator");
  return node;
}
std::string formatAttribute(const AttributeValue& value) {
  std::string result;
  llvm::raw_string_ostream out(result);
  const auto axes = [&](const patterns::Axes& list) {
    out << '[';
    for (auto axis : list) out << axis << ',';
    out << ']';
  };
  out << value.index() << ':';
  if (auto* a = std::get_if<patterns::Axes>(&value))
    axes(*a);
  else if (auto* kind = std::get_if<ReduceKind>(&value))
    out << static_cast<int>(*kind);
  else if (auto* dims = std::get_if<patterns::DotDimensions>(&value)) {
    axes(dims->lhs_contracting);
    axes(dims->rhs_contracting);
    axes(dims->lhs_batching);
    axes(dims->rhs_batching);
  } else if (auto* broadcast = std::get_if<BroadcastAttrs>(&value)) {
    axes(broadcast->dimensions);
    out << broadcast->result_type;
  } else {
    auto precision = std::get<mlir::ArrayAttr>(value);
    if (precision)
      out << precision;
    else
      out << "none";
  }
  return result;
}
}  // namespace joint_shard::semantic_detail

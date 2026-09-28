#include "research/joint_shard/bridge/tensor_lang/tensor_analysis.h"

#include <algorithm>
#include <stdexcept>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Casting.h"

namespace joint_shard {

namespace {
InferenceResult invalid(std::string reason) {
  return {InferenceStatus::Invalid, {}, std::move(reason)};
}
InferenceResult known(mlir::RankedTensorType type,
                      mlir::ElementsAttr constant = {}) {
  if (!type) return invalid("missing ranked result type");
  return {InferenceStatus::Valid, {type, constant}, {}};
}
bool axesValid(const std::vector<int64_t>& axes, int64_t rank) {
  std::vector<bool> used(rank, false);
  for (auto axis : axes) {
    if (axis < 0 || axis >= rank || used[axis]) return false;
    used[axis] = true;
  }
  return true;
}
bool compatible(int64_t a, int64_t b) {
  return mlir::ShapedType::isDynamic(a) || mlir::ShapedType::isDynamic(b) ||
         a == b;
}
bool sameShape(mlir::RankedTensorType type, const std::vector<int64_t>& shape) {
  if (type.getRank() != static_cast<int64_t>(shape.size())) return false;
  for (unsigned i = 0; i < shape.size(); ++i)
    if (!compatible(type.getDimSize(i), shape[i])) return false;
  return true;
}
bool integer(mlir::Type type) {
  auto i = llvm::dyn_cast<mlir::IntegerType>(type);
  return i && i.getWidth() > 1;
}
bool numeric(mlir::Type type) {
  return integer(type) || llvm::isa<mlir::FloatType, mlir::ComplexType>(type);
}
bool ordered(mlir::Type type) {
  return integer(type) || llvm::isa<mlir::FloatType>(type);
}
}  // namespace

bool canonicalReductionIdentity(const ReduceAttrs& attrs, mlir::Type element) {
  auto value =
      llvm::dyn_cast_or_null<mlir::DenseElementsAttr>(attrs.initializer);
  if (!value || value.getType().getRank() != 0 ||
      value.getType().getElementType() != element)
    return false;
  if (auto i = llvm::dyn_cast<mlir::IntegerType>(element)) {
    if (i.getWidth() <= 1) return false;
    const auto x = *value.getValues<llvm::APInt>().begin();
    switch (attrs.kind) {
      case ReduceKind::Sum:
        return x.isZero();
      case ReduceKind::Product:
        return x.isOne();
      case ReduceKind::Max:
        return x == (i.isUnsigned()
                         ? llvm::APInt::getMinValue(i.getWidth())
                         : llvm::APInt::getSignedMinValue(i.getWidth()));
      case ReduceKind::Min:
        return x == (i.isUnsigned()
                         ? llvm::APInt::getMaxValue(i.getWidth())
                         : llvm::APInt::getSignedMaxValue(i.getWidth()));
    }
  }
  if (llvm::isa<mlir::FloatType>(element)) {
    const auto x = *value.getValues<llvm::APFloat>().begin();
    switch (attrs.kind) {
      case ReduceKind::Sum:
        return x.isZero() && !x.isNegative();
      case ReduceKind::Product:
        return x.isExactlyValue(1.0);
      case ReduceKind::Max:
        return x.isInfinity() && x.isNegative();
      case ReduceKind::Min:
        return x.isInfinity() && !x.isNegative();
    }
  }
  return false;
}

InferenceResult inferDotResultType(const DotGeneralAttrs& attrs,
                                   std::span<const TensorFacts> operands) {
  if (operands.size() != 2) return invalid("dot requires two operands");
  if (!operands[0].type || !operands[1].type)
    return {InferenceStatus::Unknown, {}, "unknown dot operand type"};
  auto lhs = operands[0].type, rhs = operands[1].type;
  auto element = lhs.getElementType();
  if (!numeric(element) || rhs.getElementType() != element ||
      lhs.getEncoding() || rhs.getEncoding())
    return invalid("unsupported dot element types or encodings");
  if (attrs.dimensions.lhs_contracting.size() !=
          attrs.dimensions.rhs_contracting.size() ||
      attrs.dimensions.lhs_batching.size() !=
          attrs.dimensions.rhs_batching.size())
    return invalid("dot dimension list lengths differ");
  auto l = attrs.dimensions.lhs_batching, r = attrs.dimensions.rhs_batching;
  l.insert(l.end(), attrs.dimensions.lhs_contracting.begin(),
           attrs.dimensions.lhs_contracting.end());
  r.insert(r.end(), attrs.dimensions.rhs_contracting.begin(),
           attrs.dimensions.rhs_contracting.end());
  if (!axesValid(l, lhs.getRank()) || !axesValid(r, rhs.getRank()))
    return invalid("dot axes overlap or are out of range");
  for (unsigned i = 0; i < l.size(); ++i)
    if (!compatible(lhs.getDimSize(l[i]), rhs.getDimSize(r[i])))
      return invalid("dot paired dimensions differ");
  std::vector<int64_t> shape;
  for (auto axis : attrs.dimensions.lhs_batching)
    shape.push_back(lhs.getDimSize(axis));
  for (int64_t axis = 0; axis < lhs.getRank(); ++axis)
    if (std::find(l.begin(), l.end(), axis) == l.end())
      shape.push_back(lhs.getDimSize(axis));
  for (int64_t axis = 0; axis < rhs.getRank(); ++axis)
    if (std::find(r.begin(), r.end(), axis) == r.end())
      shape.push_back(rhs.getDimSize(axis));
  if (attrs.extra_attributes)
    for (auto attr : attrs.extra_attributes)
      if (attr.getName().getValue() == "dot_dimension_numbers" ||
          attr.getName().getValue() == "precision_config" ||
          attr.getName().getValue() == "algorithm")
        return invalid("dot has duplicate managed attributes");
  return known(mlir::RankedTensorType::get(shape, element));
}

mlir::ElementsAttr canonicalReductionInitializer(ReduceKind kind,
                                                 mlir::Type element) {
  auto scalar = mlir::RankedTensorType::get({}, element);
  if (auto i = llvm::dyn_cast<mlir::IntegerType>(element)) {
    if (i.getWidth() <= 1) return {};
    llvm::APInt value(i.getWidth(), kind == ReduceKind::Product ? 1 : 0);
    if (kind == ReduceKind::Max)
      value = i.isUnsigned() ? llvm::APInt::getMinValue(i.getWidth())
                             : llvm::APInt::getSignedMinValue(i.getWidth());
    if (kind == ReduceKind::Min)
      value = i.isUnsigned() ? llvm::APInt::getMaxValue(i.getWidth())
                             : llvm::APInt::getSignedMaxValue(i.getWidth());
    return mlir::DenseElementsAttr::get(scalar, value);
  }
  if (auto f = llvm::dyn_cast<mlir::FloatType>(element)) {
    llvm::APFloat value(f.getFloatSemantics());
    if (kind == ReduceKind::Product) {
      value = llvm::APFloat(1.0);
      bool loses_info = false;
      value.convert(f.getFloatSemantics(), llvm::APFloat::rmNearestTiesToEven,
                    &loses_info);
    }
    if (kind == ReduceKind::Max || kind == ReduceKind::Min)
      value =
          llvm::APFloat::getInf(f.getFloatSemantics(), kind == ReduceKind::Max);
    return mlir::DenseElementsAttr::get(scalar, value);
  }
  return {};
}

InferenceResult completeTensorNode(TensorNode& node,
                                   std::span<const TensorFacts> operands) {
  if (!validNodeSchema(node) || operands.size() != node.operands.size())
    return invalid("operator attributes or arity do not match its schema");
  if (auto* attrs = std::get_if<DotGeneralAttrs>(&node.attrs)) {
    if (!attrs->result_type) {
      auto result = inferDotResultType(*attrs, operands);
      if (!result.valid()) return result;
      attrs->result_type = result.facts.type;
    }
  }
  if (auto* attrs = std::get_if<ReduceAttrs>(&node.attrs)) {
    if (!attrs->initializer) {
      if (!operands[0].type)
        return invalid("reduce requires a known operand type");
      attrs->initializer = canonicalReductionInitializer(
          attrs->kind, operands[0].type.getElementType());
    }
  }
  return inferTensorNode(node, operands);
}

InferenceResult inferTensorNode(const TensorNode& node,
                                std::span<const TensorFacts> operands) {
  if (!validNodeSchema(node) || operands.size() != node.operands.size())
    return invalid("operator attributes or arity do not match its schema");
  if (node.op == OpKind::Input)
    return known(std::get<InputAttrs>(node.attrs).type);
  if (node.op == OpKind::Constant) {
    auto value = std::get<ConstantAttrs>(node.attrs).value;
    if (!value) return invalid("constant has no value");
    return known(llvm::dyn_cast<mlir::RankedTensorType>(value.getType()),
                 value);
  }
  for (const auto& operand : operands)
    if (!operand.type)
      return {InferenceStatus::Unknown, {}, "unknown operand type"};
  auto lhs = operands[0].type;
  auto element = lhs.getElementType();
  if (opSchema(node.op)->attribute_free) {
    for (const auto& operand : operands)
      if (operand.type != lhs)
        return invalid("elementwise operand types differ");
    bool accepted = numeric(element);
    switch (node.op) {
      case OpKind::Exp:
      case OpKind::Log:
      case OpKind::Sqrt:
      case OpKind::Tanh:
        accepted = llvm::isa<mlir::FloatType, mlir::ComplexType>(element);
        break;
      case OpKind::Maximum:
      case OpKind::Minimum:
        accepted = ordered(element);
        break;
      default:
        break;
    }
    if (!accepted) return invalid("unsupported element type for this operator");
    return known(lhs);
  }
  if (node.op == OpKind::Transpose) {
    const auto& p = std::get<TransposeAttrs>(node.attrs).permutation;
    if (p.size() != static_cast<size_t>(lhs.getRank()) ||
        !axesValid(p, lhs.getRank()))
      return invalid("transpose requires a rank-sized permutation");
    if (lhs.getEncoding())
      return invalid(
          "transpose of an encoded type requires an encoding policy");
    std::vector<int64_t> shape;
    for (auto axis : p) shape.push_back(lhs.getDimSize(axis));
    return known(mlir::RankedTensorType::get(shape, element));
  }
  if (node.op == OpKind::Reshape) {
    auto result = std::get<ReshapeAttrs>(node.attrs).result_type;
    if (!result || result.getElementType() != element ||
        result.getEncoding() != lhs.getEncoding())
      return invalid("reshape type mismatch");
    // Dynamic reshape cannot prove element conservation from types alone.
    if (!lhs.hasStaticShape() || !result.hasStaticShape())
      return {InferenceStatus::Unknown,
              {},
              "dynamic reshape element count is unknown"};
    if (lhs.getNumElements() != result.getNumElements())
      return invalid("reshape changes element count");
    return known(result);
  }
  if (node.op == OpKind::BroadcastInDim) {
    const auto& attrs = std::get<BroadcastAttrs>(node.attrs);
    auto result = attrs.result_type;
    if (!result || result.getElementType() != element ||
        result.getEncoding() != lhs.getEncoding() ||
        attrs.dimensions.size() != static_cast<size_t>(lhs.getRank()) ||
        !axesValid(attrs.dimensions, result.getRank()) ||
        !std::is_sorted(attrs.dimensions.begin(), attrs.dimensions.end()))
      return invalid("broadcast dimensions or result type are invalid");
    for (unsigned i = 0; i < attrs.dimensions.size(); ++i)
      if (lhs.getDimSize(i) != 1 &&
          !compatible(lhs.getDimSize(i),
                      result.getDimSize(attrs.dimensions[i])))
        return invalid("broadcast non-unit dimension mismatch");
    return known(result);
  }
  if (node.op == OpKind::Reduce) {
    const auto& attrs = std::get<ReduceAttrs>(node.attrs);
    if (!axesValid(attrs.axes, lhs.getRank()) ||
        !canonicalReductionIdentity(attrs, element))
      return invalid(
          "reduce requires distinct valid axes and a canonical scalar "
          "identity");
    if (lhs.getEncoding())
      return invalid("reduction encoding requires a policy");
    std::vector<int64_t> shape;
    for (int64_t i = 0; i < lhs.getRank(); ++i)
      if (std::find(attrs.axes.begin(), attrs.axes.end(), i) ==
          attrs.axes.end())
        shape.push_back(lhs.getDimSize(i));
    return known(mlir::RankedTensorType::get(shape, element));
  }
  if (node.op == OpKind::DotGeneral) {
    const auto& attrs = std::get<DotGeneralAttrs>(node.attrs);
    auto result = inferDotResultType(attrs, operands);
    if (!result.valid()) return result;
    if (!attrs.result_type || attrs.result_type.getEncoding() ||
        attrs.result_type.getElementType() != element)
      return invalid("unsupported dot result type or encoding");
    std::vector<int64_t> shape(result.facts.type.getShape().begin(),
                               result.facts.type.getShape().end());
    if (!sameShape(attrs.result_type, shape))
      return invalid("dot result shape differs from inferred shape");
    return known(attrs.result_type);
  }
  return invalid("operator inference is not implemented");
}

const TensorFacts& tensorFacts(const TensorEGraph& graph, eggc::Id id) {
  return graph.analysis_data(id);
}
TensorFacts TensorAnalysis::make(const TensorEGraph& graph,
                                 const TensorNode& node) const {
  if (!validNodeSchema(node))
    throw std::invalid_argument("operator attributes or arity mismatch");
  if (node.op == OpKind::Input && !std::get<InputAttrs>(node.attrs).type)
    throw std::invalid_argument("input requires a ranked tensor type");
  std::vector<TensorFacts> operands;
  for (auto child : node.operands)
    operands.push_back(tensorFacts(graph, child));
  auto result = inferTensorNode(node, operands);
  if (result.status == InferenceStatus::Invalid)
    throw eggc::AnalysisConflict(result.reason);
  return result.facts;
}
eggc::AnalysisMerge TensorAnalysis::merge(TensorFacts& lhs,
                                          const TensorFacts& rhs) const {
  if (lhs.type && rhs.type && lhs.type != rhs.type)
    return eggc::AnalysisMerge::Conflict;
  if (lhs.constant && rhs.constant && lhs.constant != rhs.constant)
    return eggc::AnalysisMerge::Conflict;
  bool changed = false;
  if (!lhs.type && rhs.type) {
    lhs.type = rhs.type;
    changed = true;
  }
  if (!lhs.constant && rhs.constant) {
    lhs.constant = rhs.constant;
    changed = true;
  }
  return changed ? eggc::AnalysisMerge::Changed
                 : eggc::AnalysisMerge::Unchanged;
}

}  // namespace joint_shard

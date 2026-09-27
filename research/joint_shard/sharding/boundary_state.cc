#include "research/joint_shard/sharding/boundary_state.h"

#include <algorithm>
#include <functional>
#include <stdexcept>

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Builders.h"
#include "research/joint_shard/transforms/regionizer.h"
#include "shardy/dialect/sdy/ir/utils.h"

namespace joint_shard {

namespace {
bool sameMesh(mlir::sdy::TensorShardingAttr sharding, const MeshContext& mesh) {
  if (auto ref =
          llvm::dyn_cast<mlir::FlatSymbolRefAttr>(sharding.getMeshOrRef()))
    return ref.getValue() == mesh.name;
  return sharding.getMeshOrRef() == mesh.mesh;
}
bool compatible(mlir::sdy::TensorShardingAttr layout,
                mlir::sdy::TensorShardingAttr constraint,
                const MeshContext& mesh) {
  if (!sameMesh(constraint, mesh) || constraint.getRank() != layout.getRank() ||
      !constraint.getUnreducedAxes().empty())
    return false;
  for (int64_t i = 0; i < layout.getRank(); ++i) {
    auto required = constraint.getDimSharding(i);
    auto actual = layout.getDimSharding(i).getAxes();
    auto prefix = required.getAxes();
    if (required.getIsClosed()
            ? actual != prefix
            : actual.size() < prefix.size() ||
                  !std::equal(prefix.begin(), prefix.end(), actual.begin()))
      return false;
    for (auto axis : actual)
      for (auto replicated : constraint.getReplicatedAxes())
        if (axis.overlaps(replicated)) return false;
  }
  return true;
}
mlir::sdy::TensorShardingAttr closeSharding(
    mlir::sdy::TensorShardingAttr source, const MeshContext& mesh) {
  std::vector<mlir::sdy::DimensionShardingAttr> dimensions;
  for (auto dim : source.getDimShardings())
    dimensions.push_back(mlir::sdy::DimensionShardingAttr::get(
        source.getContext(), dim.getAxes(), true));
  return mlir::sdy::TensorShardingAttr::get(source.getContext(), mesh.name,
                                            dimensions, {}, {});
}
}  // namespace

MeshContext selectMesh(mlir::ModuleOp module, std::string name) {
  module.getContext()->getOrLoadDialect<mlir::sdy::SdyDialect>();
  std::vector<MeshContext> meshes;
  for (auto op : module.getOps<mlir::sdy::MeshOp>())
    if (!op.getMesh().empty() && !op.getMesh().isMaximal())
      meshes.push_back({op.getSymName().str(), op.getMesh()});
  if (meshes.empty()) {
    if (!name.empty())
      throw std::invalid_argument("selected mesh does not exist");
    auto* context = module.getContext();
    return {"joint_mesh",
            mlir::sdy::MeshAttr::get(
                context, {mlir::sdy::MeshAxisAttr::get(context, "data", 2),
                          mlir::sdy::MeshAxisAttr::get(context, "model", 2)})};
  }
  for (const auto& mesh : meshes)
    if (mesh.name == name) return mesh;
  if (!name.empty())
    throw std::invalid_argument("selected mesh does not exist");
  if (meshes.size() != 1)
    throw std::invalid_argument("multiple meshes: select one with --mesh=name");
  return meshes.front();
}

std::string TensorSharding::str() const {
  std::string text;
  llvm::raw_string_ostream out(text);
  if (attr)
    out << attr;
  else
    out << "<missing>";
  return text;
}
std::string BoundaryState::key() const {
  std::string result = "[";
  for (const auto& input : inputs) result += input.str() + ";";
  result += "] -> [";
  for (const auto& output : outputs) result += output.str() + ";";
  return result + "]";
}
TensorSharding replicatedSharding(mlir::RankedTensorType type,
                                  const MeshContext& mesh) {
  auto* context = type.getContext();
  std::vector<mlir::sdy::DimensionShardingAttr> dimensions(
      type.getRank(), mlir::sdy::DimensionShardingAttr::get(context, {}, true));
  return {mlir::sdy::TensorShardingAttr::get(context, mesh.name, dimensions, {},
                                             {})};
}

bool validExactSharding(TensorSharding sharding, mlir::RankedTensorType type,
                        const MeshContext& mesh) {
  auto attr = sharding.attr;
  if (!type.hasStaticShape() || !attr || !attr.isFullyClosed() ||
      attr.getRank() != type.getRank() || !sameMesh(attr, mesh) ||
      !attr.getUnreducedAxes().empty())
    return false;
  std::vector<mlir::sdy::AxisRefAttr> used;
  for (int64_t i = 0; i < type.getRank(); ++i) {
    int64_t size = type.getDimSize(i);
    for (auto axis : attr.getDimSharding(i).getAxes()) {
      if (!mesh.mesh.hasAxis(axis.getName())) return false;
      if (auto sub = axis.getSubAxisInfo()) {
        auto full = mesh.mesh.getAxisSize(axis.getName());
        if (sub.getPreSize() <= 0 || sub.getSize() <= 0 ||
            full % sub.getPreSize() ||
            (full / sub.getPreSize()) % sub.getSize())
          return false;
      }
      // Exact evenly-divisible layouts only. Reject overlapping axis intervals.
      for (auto other : used)
        if (axis.overlaps(other)) return false;
      used.push_back(axis);
      auto axisSize = axis.getSize(mesh.mesh);
      if (axisSize <= 0 || size % axisSize) return false;
      size /= axisSize;
    }
  }
  for (auto axis : attr.getReplicatedAxes()) {
    if (!mesh.mesh.hasAxis(axis.getName())) return false;
    for (auto other : used)
      if (axis.overlaps(other)) return false;
  }
  return true;
}

std::vector<TensorSharding> tensorLayoutChoices(
    mlir::RankedTensorType type, const MeshContext& mesh,
    const LayoutPolicy& policy, const DimensionPolicy& dimensions,
    llvm::ArrayRef<mlir::sdy::TensorShardingAttr> constraints) {
  if (!type.hasStaticShape())
    throw std::invalid_argument(
        "boundary search requires static tensor shapes");
  std::vector<TensorSharding> choices;
  auto append = [&](TensorSharding layout) {
    if (!validExactSharding(layout, type, mesh)) return;
    for (auto constraint : constraints)
      if (!compatible(layout.attr, constraint, mesh)) return;
    if (std::find(choices.begin(), choices.end(), layout) == choices.end())
      choices.push_back(layout);
  };
  // Preserve fixed layouts outside the canonical R/DP/TP vocabulary. Closing
  // an open annotation without adding axes is also one legal refinement.
  for (auto constraint : constraints)
    if (sameMesh(constraint, mesh) && constraint.getRank() == type.getRank() &&
        constraint.getUnreducedAxes().empty())
      append({closeSharding(constraint, mesh)});
  append(replicatedSharding(type, mesh));
  auto shard = [&](const std::string& axis, std::optional<int64_t> dimension) {
    if (!dimension || !type.getRank() || !mesh.mesh.hasAxis(axis)) return;
    int64_t dim = *dimension < 0 ? type.getRank() + *dimension : *dimension;
    if (dim < 0 || dim >= type.getRank()) return;
    auto layout = replicatedSharding(type, mesh).attr;
    std::vector<mlir::sdy::DimensionShardingAttr> dims(
        layout.getDimShardings().begin(), layout.getDimShardings().end());
    dims[dim] = mlir::sdy::DimensionShardingAttr::get(
        type.getContext(),
        {mlir::sdy::AxisRefAttr::get(type.getContext(), axis)}, true);
    append({mlir::sdy::TensorShardingAttr::get(type.getContext(), mesh.name,
                                               dims, {}, {})});
  };
  shard(policy.data_axis, dimensions.data_dimension);
  shard(policy.model_axis, dimensions.model_dimension);
  return choices;
}

BoundaryEnumeration enumerateBoundaryStates(
    const Region& region, const MeshContext& mesh, const LayoutPolicy& policy,
    size_t max_states, const std::map<void*, TensorSharding>& fixed_inputs,
    const std::map<void*, TensorSharding>& fixed_outputs) {
  if (!max_states) throw std::invalid_argument("boundary cap must be positive");
  std::vector<std::vector<TensorSharding>> ports;
  auto choices = [&](mlir::Value value, size_t index, bool input) {
    std::vector<mlir::sdy::TensorShardingAttr> constraints;
    if (auto attr = mlir::sdy::getSharding(value)) constraints.push_back(attr);
    if (!input) {
      for (auto& use : value.getUses()) {
        if (auto ret = llvm::dyn_cast<mlir::func::ReturnOp>(use.getOwner())) {
          auto function = ret->getParentOfType<mlir::func::FuncOp>();
          if (auto attr =
                  function.getResultAttrOfType<mlir::sdy::TensorShardingAttr>(
                      use.getOperandNumber(), "sdy.sharding"))
            constraints.push_back(attr);
        } else if (auto constraint =
                       llvm::dyn_cast<mlir::sdy::ShardingConstraintOp>(
                           use.getOwner())) {
          // Dangling constraints apply to the source value itself. Used
          // constraints apply only to their result uses and remain boundaries.
          if (constraint->use_empty())
            constraints.push_back(constraint.getSharding());
        }
      }
    }
    const auto& overrides = input ? policy.inputs : policy.outputs;
    auto found = overrides.find(index);
    const auto& fixed = input ? fixed_inputs : fixed_outputs;
    if (auto exact = fixed.find(value.getAsOpaquePointer());
        exact != fixed.end()) {
      for (auto constraint : constraints)
        if (!compatible(exact->second.attr, constraint, mesh))
          throw std::invalid_argument(
              "function contract conflicts with region constraint");
      return std::vector<TensorSharding>{exact->second};
    }
    return tensorLayoutChoices(
        llvm::cast<mlir::RankedTensorType>(value.getType()), mesh, policy,
        found == overrides.end() ? policy.defaults : found->second,
        constraints);
  };
  for (size_t i = 0; i < region.inputs.size(); ++i)
    ports.push_back(choices(region.inputs[i], i, true));
  for (size_t i = 0; i < region.outputs.size(); ++i)
    ports.push_back(choices(region.outputs[i], i, false));
  BoundaryEnumeration result;
  // An impossible port must not cause exhaustive traversal of all preceding
  // combinations merely to discover that no complete state exists.
  if (llvm::any_of(ports, [](const auto& choices) { return choices.empty(); }))
    return result;
  std::vector<TensorSharding> selected;
  std::function<void(size_t)> visit = [&](size_t port) {
    if (result.truncated) return;
    if (port == ports.size()) {
      if (result.states.size() == max_states) {
        result.truncated = true;
        return;
      }
      BoundaryState state;
      state.inputs.assign(selected.begin(),
                          selected.begin() + region.inputs.size());
      state.outputs.assign(selected.begin() + region.inputs.size(),
                           selected.end());
      result.states.push_back(std::move(state));
      return;
    }
    for (auto layout : ports[port]) {
      selected.push_back(layout);
      visit(port + 1);
      selected.pop_back();
      if (result.truncated) break;
    }
  };
  visit(0);
  return result;
}

BoundaryState functionLayoutContract(mlir::func::FuncOp function,
                                     const MeshContext& mesh) {
  BoundaryState result;
  auto choose = [&](mlir::Type type, mlir::sdy::TensorShardingAttr annotation) {
    auto tensor = llvm::dyn_cast<mlir::RankedTensorType>(type);
    if (!tensor || !tensor.hasStaticShape())
      throw std::invalid_argument(
          "function contract requires static ranked tensor ports");
    TensorSharding layout =
        annotation ? TensorSharding{closeSharding(annotation, mesh)}
                   : replicatedSharding(tensor, mesh);
    if (!validExactSharding(layout, tensor, mesh) ||
        (annotation && !compatible(layout.attr, annotation, mesh)))
      throw std::invalid_argument(
          "no compatible exact function layout contract");
    return layout;
  };
  for (size_t i = 0; i < function.getNumArguments(); ++i)
    result.inputs.push_back(
        choose(function.getArgument(i).getType(),
               function.getArgAttrOfType<mlir::sdy::TensorShardingAttr>(
                   i, "sdy.sharding")));
  for (size_t i = 0; i < function.getNumResults(); ++i)
    result.outputs.push_back(
        choose(function.getResultTypes()[i],
               function.getResultAttrOfType<mlir::sdy::TensorShardingAttr>(
                   i, "sdy.sharding")));
  return result;
}

}  // namespace joint_shard

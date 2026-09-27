#include "research/joint_shard/sharding/plan_materializer.h"

#include <algorithm>
#include <stdexcept>

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "shardy/dialect/sdy/ir/utils.h"

namespace joint_shard {

namespace {
TensorSharding layout(mlir::Value value, const MeshContext& mesh) {
  auto attr = mlir::sdy::getSharding(value);
  return attr ? TensorSharding{attr}
              : replicatedSharding(
                    llvm::cast<mlir::RankedTensorType>(value.getType()), mesh);
}
bool close(double a, double b) {
  return std::abs(a - b) <= 1e-9 * std::max({1.0, std::abs(a), std::abs(b)});
}
}  // namespace
PlanMaterializer::PlanMaterializer(const MeshContext& mesh,
                                   llvm::ArrayRef<mlir::Type> inputs,
                                   llvm::ArrayRef<mlir::Type> outputs,
                                   const BoundaryState& boundary)
    : mesh_(mesh), boundary_(boundary), builder_(mesh.mesh.getContext()) {
  if (inputs.size() != boundary.inputs.size() ||
      outputs.size() != boundary.outputs.size())
    throw std::invalid_argument("materializer boundary arity mismatch");
  auto loc = builder_.getUnknownLoc();
  module_ = mlir::ModuleOp::create(loc);
  builder_.setInsertionPointToStart(module_->getBody());
  builder_.create<mlir::sdy::MeshOp>(loc, mesh.name, mesh.mesh);
  function_ = builder_.create<mlir::func::FuncOp>(
      loc, "main", builder_.getFunctionType(inputs, outputs));
  function_.addEntryBlock();
  for (size_t i = 0; i < inputs.size(); ++i)
    function_.setArgAttr(i, "sdy.sharding", boundary.inputs[i].attr);
  for (size_t i = 0; i < outputs.size(); ++i)
    function_.setResultAttr(i, "sdy.sharding", boundary.outputs[i].attr);
  builder_.setInsertionPointToStart(&function_.getBody().front());
}
std::vector<mlir::Value> PlanMaterializer::inlineArtifact(
    const std::string& artifact, mlir::ValueRange inputs) {
  auto source =
      mlir::parseSourceString<mlir::ModuleOp>(artifact, builder_.getContext());
  if (!source || mlir::failed(mlir::verify(*source)))
    throw std::invalid_argument("invalid lowered artifact");
  size_t meshes = 0, functions = 0;
  for (auto& op : source->getBody()->getOperations()) {
    if (auto mesh = llvm::dyn_cast<mlir::sdy::MeshOp>(op)) {
      ++meshes;
      if (mesh.getSymName() != mesh_.name || mesh.getMesh() != mesh_.mesh)
        throw std::invalid_argument("artifact mesh mismatch");
    } else if (llvm::isa<mlir::func::FuncOp>(op))
      ++functions;
    else
      throw std::invalid_argument("unsupported artifact top-level operation");
  }
  auto fn = source->lookupSymbol<mlir::func::FuncOp>("main");
  if (meshes != 1 || functions != 1 || !fn || fn.isExternal() ||
      !fn.getBody().hasOneBlock() || fn.getNumArguments() != inputs.size())
    throw std::invalid_argument(
        "artifact must have one mesh and single-block main");
  mlir::IRMapping mapping;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i].getType() != fn.getArgument(i).getType() ||
        layout(inputs[i], mesh_) != layout(fn.getArgument(i), mesh_))
      throw std::invalid_argument("artifact input type/layout mismatch");
    mapping.map(fn.getArgument(i), inputs[i]);
  }
  for (auto& op : fn.getBody().front().without_terminator()) {
    auto* clone = builder_.clone(op, mapping);
    // Missing internal post-propagation attrs mean replication. Make this
    // explicit so inlining cannot accidentally inherit a different layout.
    for (auto result : clone->getResults())
      if (llvm::isa<mlir::RankedTensorType>(result.getType()) &&
          !mlir::sdy::getSharding(result))
        mlir::sdy::setSharding(
            result,
            replicatedSharding(
                llvm::cast<mlir::RankedTensorType>(result.getType()), mesh_)
                .attr);
  }
  auto ret = llvm::dyn_cast<mlir::func::ReturnOp>(
      fn.getBody().front().getTerminator());
  if (!ret) throw std::invalid_argument("artifact missing return");
  std::vector<mlir::Value> results;
  for (size_t i = 0; i < ret.getNumOperands(); ++i) {
    auto value = mapping.lookup(ret.getOperand(i));
    auto contract = fn.getResultAttrOfType<mlir::sdy::TensorShardingAttr>(
        i, "sdy.sharding");
    if (!contract || layout(value, mesh_).attr != contract)
      throw std::invalid_argument("artifact output layout mismatch");
    results.push_back(value);
  }
  return results;
}
std::string PlanMaterializer::finish(mlir::ValueRange outputs,
                                     const Cost& expected,
                                     const CostModel& model) {
  if (outputs.size() != boundary_.outputs.size())
    throw std::invalid_argument("materializer output arity mismatch");
  for (size_t i = 0; i < outputs.size(); ++i)
    if (outputs[i].getType() != function_.getResultTypes()[i] ||
        layout(outputs[i], mesh_) != boundary_.outputs[i])
      throw std::invalid_argument("materializer output contract mismatch");
  builder_.create<mlir::func::ReturnOp>(builder_.getUnknownLoc(), outputs);
  if (mlir::failed(mlir::verify(*module_)))
    throw std::runtime_error("composed module failed verification");
  auto actual = model.estimate(*module_, mesh_.mesh);
  if (!expected.known() || !actual.known() ||
      !close(actual.compute, expected.compute) ||
      !close(actual.communication, expected.communication))
    throw std::runtime_error(
        "materialized cost differs from selected component costs");
  std::string text;
  llvm::raw_string_ostream out(text);
  module_->print(out);
  return text;
}

}  // namespace joint_shard

#include "research/joint_shard/sharding/region_evaluator.h"

#include <stdexcept>

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "research/joint_shard/bridge/stablehlo_exporter.h"
#include "research/joint_shard/transforms/region_candidates.h"
#include "shardy/dialect/sdy/ir/utils.h"

namespace joint_shard {

namespace {
mlir::OwningOpRef<mlir::ModuleOp> createWrapper(
    mlir::Location location, const MeshContext& mesh,
    llvm::ArrayRef<mlir::Type> inputs, llvm::ArrayRef<mlir::Type> outputs) {
  auto module = mlir::ModuleOp::create(location);
  mlir::OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBody());
  builder.create<mlir::sdy::MeshOp>(location, mesh.name, mesh.mesh);
  auto function = builder.create<mlir::func::FuncOp>(
      location, "main", builder.getFunctionType(inputs, outputs));
  function.addEntryBlock();
  return module;
}
bool matchesBoundary(mlir::func::FuncOp function, const BoundaryState& state) {
  for (size_t i = 0; i < state.inputs.size(); ++i)
    if (function.getArgAttr(i, "sdy.sharding") != state.inputs[i].attr)
      return false;
  for (size_t i = 0; i < state.outputs.size(); ++i)
    if (function.getResultAttr(i, "sdy.sharding") != state.outputs[i].attr)
      return false;
  return true;
}
}  // namespace

mlir::OwningOpRef<mlir::ModuleOp> prepareCandidateModule(
    const Region& region, const Candidate& candidate, const MeshContext& mesh) {
  if (region.operations.empty()) throw std::invalid_argument("empty region");
  if (candidate.output_roots.size() != region.outputs.size())
    throw std::invalid_argument("candidate output arity mismatch");
  std::vector<mlir::Type> inputs, outputs;
  for (auto value : region.inputs) inputs.push_back(value.getType());
  for (auto value : region.outputs) outputs.push_back(value.getType());
  auto location = region.operations.front()->getLoc();
  auto module = createWrapper(location, mesh, inputs, outputs);
  auto function = module->lookupSymbol<mlir::func::FuncOp>("main");
  mlir::OpBuilder builder(module->getContext());
  builder.setInsertionPointToStart(&function.getBody().front());
  std::vector<mlir::Value> arguments(function.getArguments().begin(),
                                     function.getArguments().end());
  StableHloExporter exporter(builder, location, std::move(arguments));
  auto values =
      exporter.exportRoots(candidate.expression, candidate.output_roots);
  builder.create<mlir::func::ReturnOp>(location, values);
  if (mlir::failed(mlir::verify(*module)))
    throw std::invalid_argument("candidate export failed verification");
  return module;
}

EvaluationResult RegionEvaluator::evaluate(
    mlir::ModuleOp prepared, const BoundaryState& boundary,
    const ShardyRunOptions& options) const {
  EvaluationResult result;
  if (options.stop_after != ShardyStage::Collectives)
    throw std::invalid_argument(
        "region evaluation requires collective conversion");
  mlir::OwningOpRef<mlir::ModuleOp> module(prepared.clone());
  auto function = module->lookupSymbol<mlir::func::FuncOp>("main");
  if (!function || function.getNumArguments() != boundary.inputs.size() ||
      function.getNumResults() != boundary.outputs.size()) {
    result.failure = "boundary arity mismatch";
    return result;
  }
  for (size_t i = 0; i < boundary.inputs.size(); ++i) {
    auto type = llvm::dyn_cast<mlir::RankedTensorType>(
        function.getArgument(i).getType());
    if (!type || !validExactSharding(boundary.inputs[i], type, mesh_)) {
      result.failure = "invalid input layout";
      return result;
    }
    function.setArgAttr(i, "sdy.sharding", boundary.inputs[i].attr);
  }
  for (size_t i = 0; i < boundary.outputs.size(); ++i) {
    auto type =
        llvm::dyn_cast<mlir::RankedTensorType>(function.getResultTypes()[i]);
    if (!type || !validExactSharding(boundary.outputs[i], type, mesh_)) {
      result.failure = "invalid output layout";
      return result;
    }
    function.setResultAttr(i, "sdy.sharding", boundary.outputs[i].attr);
  }
  ShardyRunner runner;
  if (mlir::failed(runner.run(*module, options))) {
    result.failure = "Shardy pipeline failed";
    return result;
  }
  function = module->lookupSymbol<mlir::func::FuncOp>("main");
  if (!matchesBoundary(function, boundary)) {
    result.failure = "Shardy changed the requested boundary contract";
    return result;
  }
  result.feasible = true;
  result.cost = model_.estimate(*module, mesh_.mesh);
  result.lowered_mlir = runner.takeFinalMlir();
  return result;
}

ReshardPlan ReshardCostOracle::plan(const TensorSharding& from,
                                    const TensorSharding& to, mlir::Type type) {
  ReshardPlan result;
  result.from = from;
  result.to = to;
  result.type = type;
  auto tensor = llvm::dyn_cast<mlir::RankedTensorType>(type);
  if (!tensor || !validExactSharding(from, tensor, mesh_) ||
      !validExactSharding(to, tensor, mesh_))
    return result;
  if (from == to) {
    result.feasible = true;
    result.cost = {};
    return result;
  }
  std::string key;
  llvm::raw_string_ostream out(key);
  out << type << "|" << from.str() << "|" << to.str();
  if (auto found = cache_.find(key); found != cache_.end())
    return found->second;
  auto location = mlir::UnknownLoc::get(type.getContext());
  auto module = createWrapper(location, mesh_, {type}, {type});
  auto function = module->lookupSymbol<mlir::func::FuncOp>("main");
  function.setArgAttr(0, "sdy.sharding", from.attr);
  function.setResultAttr(0, "sdy.sharding", to.attr);
  mlir::OpBuilder builder(type.getContext());
  builder.setInsertionPointToStart(&function.getBody().front());
  auto adapter = builder.create<mlir::sdy::ReshardOp>(
      location, function.getArgument(0), to.attr);
  builder.create<mlir::func::ReturnOp>(location, adapter.getResult());
  if (mlir::succeeded(mlir::verify(*module)) &&
      mlir::succeeded(lowerReshardsToCollectives(*module)) &&
      mlir::succeeded(mlir::verify(*module))) {
    result.cost = model_.estimate(*module, mesh_.mesh);
    result.feasible = result.cost.known();
    llvm::raw_string_ostream text(result.lowered_mlir);
    module->print(text);
  }
  return cache_[key] = result;
}
Cost ReshardCostOracle::estimate(const TensorSharding& from,
                                 const TensorSharding& to, mlir::Type type) {
  return plan(from, to, type).cost;
}

}  // namespace joint_shard

#include "research/joint_shard/transforms/rewrite_regions.h"

#include <limits>
#include <optional>
#include <vector>

#include "llvm/ADT/DenseSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "eggc/egraph.hpp"
#include "eggc/extract.hpp"
#include "eggc/parser.hpp"
#include "eggc/runner.hpp"
#include "research/joint_shard/bridge/operation_descriptors.h"
#include "research/joint_shard/bridge/stablehlo_exporter.h"
#include "research/joint_shard/bridge/stablehlo_importer.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace {

bool isRewritable(mlir::Operation* op) {
  if (!op || op->getNumRegions() || op->getNumResults() != 1) return false;
  // Treat every annotated result as a boundary, including partially open ones.
  if (op->hasAttr("sdy.sharding")) return false;
  if (llvm::isa<mlir::stablehlo::AddOp, mlir::stablehlo::MulOp,
                mlir::stablehlo::ExpOp>(op)) {
    // These exporter paths cannot preserve arbitrary operation attributes.
    return op->getAttrs().empty();
  }
  auto dot = llvm::dyn_cast<mlir::stablehlo::DotGeneralOp>(op);
  if (!dot) return false;
  auto lhs = llvm::dyn_cast<mlir::RankedTensorType>(dot.getLhs().getType());
  auto rhs = llvm::dyn_cast<mlir::RankedTensorType>(dot.getRhs().getType());
  auto dims = dot.getDotDimensionNumbers();
  return lhs && rhs && lhs.getRank() == 2 && rhs.getRank() == 2 &&
         dims.getLhsBatchingDimensions().empty() &&
         dims.getRhsBatchingDimensions().empty() &&
         dims.getLhsContractingDimensions().size() == 1 &&
         dims.getRhsContractingDimensions().size() == 1 &&
         dims.getLhsContractingDimensions()[0] == 1 &&
         dims.getRhsContractingDimensions()[0] == 0;
}

mlir::Value rewriteRoot(mlir::Value root, mlir::Operation* consumer) {
  if (!isRewritable(root.getDefiningOp())) return root;
  std::vector<mlir::Value> leaves;
  llvm::DenseSet<mlir::Value> visited;
  const auto collect = [&](auto&& self, mlir::Value value) -> void {
    if (!visited.insert(value).second) return;
    if (!isRewritable(value.getDefiningOp())) {
      leaves.push_back(value);
      return;
    }
    for (mlir::Value operand : value.getDefiningOp()->getOperands()) {
      self(self, operand);
    }
  };
  collect(collect, root);

  OperationDescriptors descriptors;
  eggc::EGraph graph;
  StableHloImporter importer(graph, &descriptors);
  for (unsigned i = 0; i < leaves.size(); ++i) importer.bindValue(leaves[i], i);
  eggc::Id id = importer.importValue(root);
  eggc::run(graph,
            {eggc::parse_rewrite("commute-add", "(add ?x ?y)", "(add ?y ?x)")});
  // Demonstration extraction: prefer a smaller subtree on the left of add.
  auto sizeCost = eggc::ast_size_cost();
  auto cost = [sizeCost](const eggc::ENode& node,
                         const std::vector<std::size_t>& children)
      -> std::optional<std::size_t> {
    auto result = sizeCost(node, children);
    if (result && node.op == "add" && children.size() == 2 &&
        children[0] > children[1]) {
      if (*result == std::numeric_limits<std::size_t>::max())
        return std::nullopt;
      ++*result;
    }
    return result;
  };
  auto expression = eggc::Extractor(graph, cost).find_best_rec_expr(id).second;
  mlir::OpBuilder builder(consumer);
  StableHloExporter exporter(builder, consumer->getLoc(), std::move(leaves),
                             &descriptors);
  return exporter.exportExpr(expression, expression.root().value);
}

}  // namespace

mlir::LogicalResult rewriteUnconstrainedRegions(mlir::ModuleOp module) {
  // Validate the supported function structure before mutating any function.
  for (auto function : module.getOps<mlir::func::FuncOp>()) {
    if (!function.isExternal() && !llvm::hasSingleElement(function.getBody())) {
      return function.emitError(
          "region rewriting requires single-block functions");
    }
  }
  for (auto function : module.getOps<mlir::func::FuncOp>()) {
    if (function.isExternal()) continue;
    std::vector<mlir::Operation*> originalOps;
    for (auto& op : function.getBody().front()) originalOps.push_back(&op);
    // Every preserved operation is a boundary, even when it is dangling.
    // Process in program order so new roots use already-rewritten boundaries.
    for (auto* op : originalOps) {
      if (isRewritable(op)) continue;
      for (auto& operand : op->getOpOperands()) {
        operand.set(rewriteRoot(operand.get(), op));
      }
    }
    for (auto it = originalOps.rbegin(); it != originalOps.rend(); ++it) {
      if (isRewritable(*it) && (*it)->use_empty()) (*it)->erase();
    }
  }
  return mlir::verify(module);
}

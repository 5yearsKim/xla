#include <algorithm>
#include <iostream>
#include <optional>
#include <unordered_set>
#include <utility>
#include <vector>

#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "eggc/egraph.hpp"
#include "eggc/expr.hpp"
#include "eggc/extract.hpp"
#include "eggc/parser.hpp"
#include "eggc/runner.hpp"
#include "research/joint_shard/stablehlo_exporter.h"
#include "research/joint_shard/stablehlo_importer.h"
#include "stablehlo/dialect/Register.h"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: parse_stablehlo <file.mlir>\n";
    return 1;
  }

  llvm::outs() << "Parsing StableHLO file: " << argv[1] << "\n";

  mlir::DialectRegistry registry;

  // Registers StableHLO and the dialects needed by StableHLO.
  mlir::stablehlo::registerAllDialects(registry);

  mlir::MLIRContext context(registry);

  auto module = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);

  if (!module) {
    std::cerr << "Failed to parse MLIR file\n";
    return 1;
  }

  auto function = module->lookupSymbol<mlir::func::FuncOp>("main");
  if (!function || function.getBody().empty()) {
    std::cerr << "Expected a defined func.func @main\n";
    return 1;
  }

  auto returnOp = llvm::dyn_cast<mlir::func::ReturnOp>(
      function.getBody().front().getTerminator());
  if (!returnOp || returnOp.getNumOperands() != 1) {
    std::cerr << "Expected @main to return exactly one value\n";
    return 1;
  }

  try {
    eggc::EGraph graph;
    OperationDescriptors descriptors;
    StableHloImporter importer(graph, &descriptors);
    mlir::Value original_root = returnOp.getOperand(0);
    eggc::Id root = importer.importValue(original_root);
    llvm::outs() << "Imported root eclass: " << root << "\n\n";

    const std::vector<eggc::Rewrite> rewrites{
        eggc::parse_rewrite("commute-add", "(add ?x ?y)", "(add ?y ?x)"),
    };
    eggc::run(graph, rewrites);

    const auto prefer_arg2_on_add_lhs =
        [&graph](const eggc::ENode& node,
                 const std::vector<std::size_t>& child_costs)
        -> std::optional<std::size_t> {
      std::size_t cost = 1;
      for (std::size_t child_cost : child_costs) cost += child_cost;
      if (node.op == "add" && node.children.size() == 2) {
        const auto& lhs_nodes = graph.nodes(graph.find(node.children[0]));
        const bool lhs_is_arg2 = std::any_of(
            lhs_nodes.begin(), lhs_nodes.end(), [](const eggc::ENode& lhs) {
              return lhs.op == "arg2" && lhs.children.empty();
            });
        if (!lhs_is_arg2) cost += 10;
      }
      return cost;
    };
    const auto [cost, expression] =
        eggc::Extractor(graph, prefer_arg2_on_add_lhs).find_best_rec_expr(root);
    (void)cost;
    llvm::outs() << "Extracted expression:\n"
                 << eggc::to_string(expression) << "\n\n";

    mlir::Block& entry = function.getBody().front();
    std::vector<mlir::Value> arguments(entry.getArguments().begin(),
                                       entry.getArguments().end());
    mlir::OpBuilder builder(returnOp);
    StableHloExporter exporter(builder, returnOp.getLoc(), std::move(arguments),
                               &descriptors);
    mlir::Value exported_root =
        exporter.exportExpr(expression, expression.root().value);
    returnOp.setOperand(0, exported_root);

    std::vector<mlir::Operation*> original_ops;
    std::unordered_set<mlir::Operation*> visited;
    const auto collect_original_ops = [&](auto&& self,
                                          mlir::Value value) -> void {
      mlir::Operation* defining_op = value.getDefiningOp();
      if (!defining_op || !visited.insert(defining_op).second) return;
      original_ops.push_back(defining_op);
      for (mlir::Value operand : defining_op->getOperands()) {
        self(self, operand);
      }
    };
    collect_original_ops(collect_original_ops, original_root);
    for (mlir::Operation* op : original_ops) {
      if (op->use_empty()) op->erase();
    }

    if (mlir::failed(mlir::verify(*module))) {
      llvm::errs() << "Generated invalid StableHLO\n";
      return 1;
    }

    llvm::outs() << "Verified exported module:\n";
    module->print(llvm::outs());
    llvm::outs() << "\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }

  return 0;
}

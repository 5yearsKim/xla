#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "eggc/egraph.hpp"
#include "eggc/extract.hpp"
#include "eggc/parser.hpp"
#include "eggc/runner.hpp"
#include "research/joint_shard/shardy_runner.h"
#include "research/joint_shard/stablehlo_exporter.h"
#include "research/joint_shard/stablehlo_importer.h"
#include "shardy/dialect/sdy/ir/register.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace {

std::vector<BoundarySharding> boundaries(const std::string& name) {
  using Kind = BoundarySharding::Kind;
  TensorSharding data{{{"data"}, {}}};
  TensorSharding model{{{}, {"model"}}};
  TensorSharding both{{{"data"}, {"model"}}};
  if (name == "compatible") {
    return {{"main", Kind::Argument, 0, data},
            {"main", Kind::Argument, 1, model},
            {"main", Kind::Result, 0, both}};
  }
  if (name == "gather") {
    return {{"main", Kind::Argument, 0, data},
            {"main", Kind::Argument, 1, model},
            {"main", Kind::Result, 0, data}};
  }
  if (name == "contracting") {
    return {{"main", Kind::Argument, 0, both},
            {"main", Kind::Argument, 1, {{{"model"}, {}}}},
            {"main", Kind::Result, 0, data}};
  }
  if (name == "elementwise") {
    return {{"main", Kind::Argument, 0, data},
            {"main", Kind::Argument, 1, data},
            {"main", Kind::Argument, 2, data},
            {"main", Kind::Result, 0, data}};
  }
  throw std::invalid_argument("unknown sharding case: " + name);
}

mlir::LogicalResult roundTrip(mlir::ModuleOp module) {
  auto function = module.lookupSymbol<mlir::func::FuncOp>("main");
  if (!function || function.isExternal() ||
      !llvm::hasSingleElement(function.getBody())) {
    return module.emitError("round trip requires a single-block @main");
  }
  auto terminator = llvm::dyn_cast<mlir::func::ReturnOp>(
      function.getBody().front().getTerminator());
  if (!terminator || terminator.getNumOperands() != 1) {
    return function.emitError("round trip requires one returned value");
  }
  OperationDescriptors descriptors;
  eggc::EGraph graph;
  StableHloImporter importer(graph, &descriptors);
  eggc::Id root = importer.importValue(terminator.getOperand(0));
  graph.rebuild();
  llvm::outs() << "Imported expression: "
               << eggc::to_string(
                      eggc::Extractor(graph).find_best_rec_expr(root).second)
               << "\n";
  eggc::run(graph,
            {eggc::parse_rewrite("commute-add", "(add ?x ?y)", "(add ?y ?x)")});
  // Prefer the smaller subtree on the left of an add to select the rewritten
  // add/multiply example. This is a demonstration cost, not a communication
  // cost.
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
  auto expression =
      eggc::Extractor(graph, cost).find_best_rec_expr(root).second;
  llvm::outs() << "Extracted expression: " << eggc::to_string(expression)
               << "\n";

  std::vector<mlir::Operation*> originalOps;
  for (auto& op : function.getBody().front()) originalOps.push_back(&op);
  auto arguments = function.getArguments();
  mlir::OpBuilder builder(terminator);
  StableHloExporter exporter(builder, terminator.getLoc(),
                             {arguments.begin(), arguments.end()},
                             &descriptors);
  terminator.setOperand(
      0, exporter.exportExpr(expression, expression.root().value));
  for (auto it = originalOps.rbegin(); it != originalOps.rend(); ++it) {
    auto* op = *it;
    if (llvm::isa<mlir::stablehlo::AddOp, mlir::stablehlo::MulOp,
                  mlir::stablehlo::ExpOp, mlir::stablehlo::DotGeneralOp>(op) &&
        op->use_empty()) {
      op->erase();
    }
  }
  return mlir::verify(module);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    llvm::errs()
        << "Usage: run_shardy <file.mlir> "
           "[--case=compatible|gather|contracting|elementwise|all] "
           "[--round-trip] [--stop-after=propagation|reshards|collectives] "
           "[--dump-dir=directory]\n";
    return 1;
  }
  try {
    std::string caseName = "compatible";
    bool roundTripEnabled = false;
    ShardyRunOptions options;
    for (int i = 2; i < argc; ++i) {
      std::string argument = argv[i];
      if (argument == "--round-trip")
        roundTripEnabled = true;
      else if (argument.compare(0, 7, "--case=") == 0)
        caseName = argument.substr(7);
      else if (argument.compare(0, 11, "--dump-dir=") == 0) {
        options.dumpDirectory = argument.substr(11);
      } else if (argument.compare(0, 13, "--stop-after=") == 0) {
        std::string stage = argument.substr(13);
        if (stage == "propagation")
          options.stopAfter = ShardyStage::Propagation;
        else if (stage == "reshards")
          options.stopAfter = ShardyStage::ExplicitReshards;
        else if (stage == "collectives")
          options.stopAfter = ShardyStage::Collectives;
        else
          throw std::invalid_argument("unknown stop stage: " + stage);
      } else {
        throw std::invalid_argument("unknown option: " + argument);
      }
    }
    std::vector<std::string> cases =
        caseName == "all"
            ? std::vector<std::string>{"compatible", "gather", "contracting"}
            : std::vector<std::string>{caseName};
    for (const auto& name : cases) (void)boundaries(name);

    mlir::DialectRegistry registry;
    mlir::stablehlo::registerAllDialects(registry);
    mlir::sdy::registerAllDialects(registry);
    mlir::MLIRContext context(registry);
    auto original = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
    if (!original || mlir::failed(mlir::verify(*original))) return 1;

    for (const std::string& name : cases) {
      mlir::OwningOpRef<mlir::ModuleOp> module(original->clone());
      llvm::outs() << "Case: " << name << "\n";
      if (roundTripEnabled && mlir::failed(roundTrip(*module))) return 1;
      auto function = module->lookupSymbol<mlir::func::FuncOp>("main");
      unsigned expectedArgs = name == "elementwise" ? 3 : 2;
      if (!function || function.getNumArguments() != expectedArgs ||
          function.getNumResults() != 1) {
        module->emitError("case requires @main with ")
            << expectedArgs << " arguments and one result";
        return 1;
      }
      ShardyRunOptions caseOptions = options;
      if (cases.size() > 1 && !caseOptions.dumpDirectory.empty()) {
        caseOptions.dumpDirectory += "/" + name;
      }
      ShardyRunner runner;
      if (mlir::failed(runner.run(*module, boundaries(name), caseOptions)))
        return 1;
      for (const auto& snapshot : runner.snapshots()) {
        unsigned reshards = 0;
        unsigned collectives = 0;
        for (const auto& [op, count] : snapshot.communicationOps) {
          if (op == "sdy.reshard")
            reshards += count;
          else
            collectives += count;
        }
        llvm::outs() << snapshot.name << ": reshards=" << reshards
                     << ", collectives=" << collectives;
        for (const auto& [op, count] : snapshot.communicationOps) {
          llvm::outs() << ", " << op << "=" << count;
        }
        llvm::outs() << "\n";
      }
      llvm::outs() << "Verified resulting module:\n"
                   << runner.snapshots().back().mlir;
    }
  } catch (const std::exception& error) {
    llvm::errs() << error.what() << "\n";
    return 1;
  }
  return 0;
}

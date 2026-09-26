#include "research/joint_shard/stablehlo_exporter.h"

#include <stdexcept>
#include <string>
#include <utility>

#include "stablehlo/dialect/StablehloOps.h"

mlir::Value StableHloExporter::exportExpr(const eggc::RecExpr& expr,
                                          std::size_t node) {
  values_.assign(expr.nodes.size(), mlir::Value());
  if (node >= expr.nodes.size()) {
    throw std::out_of_range("expression node index is out of range");
  }
  return exportNode(expr, node);
}

mlir::Value StableHloExporter::exportNode(const eggc::RecExpr& expr,
                                          std::size_t node_index) {
  if (node_index >= expr.nodes.size()) {
    throw std::out_of_range("expression child index is out of range");
  }
  if (values_[node_index]) return values_[node_index];

  const eggc::ExprNode& node = expr.nodes[node_index];
  const auto child = [&](std::size_t index) {
    if (index >= node.children.size()) {
      throw std::invalid_argument("expression node has too few children");
    }
    const std::size_t child_index = node.children[index].value;
    if (child_index >= node_index) {
      throw std::invalid_argument(
          "expression children must precede their parent");
    }
    return exportNode(expr, child_index);
  };

  if (node.op.compare(0, 3, "arg") == 0) {
    if (!node.children.empty()) {
      throw std::invalid_argument("argument expression cannot have children");
    }
    const std::string index_text = node.op.substr(3);
    std::size_t parsed = 0;
    std::size_t argument_index;
    try {
      argument_index = std::stoul(index_text, &parsed);
    } catch (const std::exception&) {
      throw std::invalid_argument("invalid function argument expression");
    }
    if (parsed != index_text.size() || argument_index >= arguments_.size()) {
      throw std::out_of_range("function argument index is out of range");
    }
    values_[node_index] = arguments_[argument_index];
    return values_[node_index];
  }

  if (node.op.compare(0, 12, "dot_general#") == 0) {
    if (!descriptors_ || node.children.size() != 2) {
      throw std::invalid_argument(
          "dot_general requires two children and descriptors");
    }
    const auto& descriptor = descriptors_->getDot(node.op);
    mlir::Value lhs = child(0);
    mlir::Value rhs = child(1);
    if (lhs.getType() != descriptor.lhsType ||
        rhs.getType() != descriptor.rhsType) {
      throw std::invalid_argument(
          "dot_general operand types changed during extraction");
    }
    mlir::OperationState state(loc_, "stablehlo.dot_general");
    state.addOperands({lhs, rhs});
    state.addTypes(descriptor.resultType);
    state.addAttributes(descriptor.attributes.getValue());
    auto dot =
        llvm::cast<mlir::stablehlo::DotGeneralOp>(builder_.create(state));
    values_[node_index] = dot.getResult();
  } else if (node.op == "add") {
    if (node.children.size() != 2) {
      throw std::invalid_argument("add expression must have two children");
    }
    auto add =
        mlir::stablehlo::AddOp::create(builder_, loc_, child(0), child(1));
    values_[node_index] = add.getResult();
  } else if (node.op == "multiply") {
    if (node.children.size() != 2) {
      throw std::invalid_argument("multiply expression must have two children");
    }
    auto multiply =
        mlir::stablehlo::MulOp::create(builder_, loc_, child(0), child(1));
    values_[node_index] = multiply.getResult();
  } else if (node.op == "exp") {
    if (node.children.size() != 1) {
      throw std::invalid_argument("exp expression must have one child");
    }
    auto exp = mlir::stablehlo::ExpOp::create(builder_, loc_, child(0));
    values_[node_index] = exp.getResult();
  } else {
    throw std::invalid_argument("unsupported egg-c expression op: " + node.op);
  }

  return values_[node_index];
}

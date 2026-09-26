#ifndef RESEARCH_JOINT_SHARD_BRIDGE_STABLEHLO_EXPORTER_H_
#define RESEARCH_JOINT_SHARD_BRIDGE_STABLEHLO_EXPORTER_H_

#include <cstddef>
#include <utility>
#include <vector>

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "eggc/expr.hpp"
#include "research/joint_shard/bridge/tensor_lang/tensorlang.h"

class StableHloExporter {
 public:
  StableHloExporter(mlir::OpBuilder& builder, mlir::Location loc,
                    std::vector<mlir::Value> inputs)
      : builder_(builder), loc_(loc), inputs_(std::move(inputs)) {}

  // Export all region outputs with a shared expression/value cache.
  std::vector<mlir::Value> exportRoots(const TensorRecExpr& expr,
                                       const std::vector<std::size_t>& roots);
  mlir::Value exportExpr(const TensorRecExpr& expr, std::size_t node);

 private:
  mlir::Value exportNode(const TensorRecExpr& expr, std::size_t node);

  mlir::OpBuilder& builder_;
  mlir::Location loc_;
  std::vector<mlir::Value> inputs_;
  std::vector<mlir::Value> values_;
  std::vector<mlir::RankedTensorType> types_;
};

#endif  // RESEARCH_JOINT_SHARD_BRIDGE_STABLEHLO_EXPORTER_H_

#ifndef RESEARCH_JOINT_SHARD_BRIDGE_STABLEHLO_EXPORTER_H_
#define RESEARCH_JOINT_SHARD_BRIDGE_STABLEHLO_EXPORTER_H_

#include <cstddef>
#include <utility>
#include <vector>

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "eggc/expr.hpp"
#include "research/joint_shard/bridge/operation_descriptors.h"

class StableHloExporter {
 public:
  StableHloExporter(mlir::OpBuilder& builder, mlir::Location loc,
                    std::vector<mlir::Value> inputs,
                    OperationDescriptors* descriptors = nullptr)
      : builder_(builder),
        loc_(loc),
        inputs_(std::move(inputs)),
        descriptors_(descriptors) {}

  mlir::Value exportExpr(const eggc::RecExpr& expr, std::size_t node);

 private:
  mlir::Value exportNode(const eggc::RecExpr& expr, std::size_t node);

  mlir::OpBuilder& builder_;
  mlir::Location loc_;
  std::vector<mlir::Value> inputs_;
  std::vector<mlir::Value> values_;
  OperationDescriptors* descriptors_;
};

#endif  // RESEARCH_JOINT_SHARD_BRIDGE_STABLEHLO_EXPORTER_H_

#ifndef RESEARCH_JOINT_SHARD_TRANSFORMS_REGIONIZER_H_
#define RESEARCH_JOINT_SHARD_TRANSFORMS_REGIONIZER_H_

#include <cstddef>
#include <string>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

// A borrowed view: the source function must remain alive and unmodified while
// its regions are imported/evaluated. Interfaces have deterministic SSA order.
struct Region {
  size_t id = 0;
  std::vector<mlir::Operation*> operations;
  std::vector<mlir::Value> inputs;
  std::vector<mlir::Value> outputs;
  bool oversized = false;
};

struct RegionizerOptions {
  size_t max_region_ops = 128;
  // Supported blockers are singleton regions, so their work is still costed.
  std::vector<std::string> rule_blockers = {"stablehlo.exponential",
                                            "stablehlo.tanh"};
};

Region describeRegion(llvm::ArrayRef<mlir::Operation*> operations);
size_t countCrossingValues(llvm::ArrayRef<mlir::Operation*> operations,
                           size_t cut);
bool cutBreaksProtectedPattern(llvm::ArrayRef<mlir::Operation*> operations,
                               size_t cut);

class Regionizer {
 public:
  explicit Regionizer(RegionizerOptions options = {}) : options_(options) {}
  std::vector<Region> split(mlir::func::FuncOp function) const;

 private:
  RegionizerOptions options_;
};

#endif

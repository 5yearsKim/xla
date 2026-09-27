#ifndef RESEARCH_JOINT_SHARD_SHARDING_COST_MODEL_H_
#define RESEARCH_JOINT_SHARD_SHARDING_COST_MODEL_H_

#include <cmath>

#include "mlir/IR/BuiltinOps.h"
#include "shardy/dialect/sdy/ir/dialect.h"

// All costs are estimated microseconds, summed sequentially without overlap.
struct Cost {
  double compute = 0;
  double communication = 0;
  unsigned unknown = 0;
  double total() const { return compute + communication; }
  bool known() const {
    return !unknown && std::isfinite(total()) && compute >= 0 &&
           communication >= 0;
  }
  Cost& operator+=(const Cost& other) {
    compute += other.compute;
    communication += other.communication;
    unknown += other.unknown;
    return *this;
  }
};

struct CostModelOptions {
  // Illustrative parameters, not a calibration for a particular device.
  double compute_work_per_us = 1000000;
  double bandwidth_bytes_per_us = 50000;
  double collective_latency_us = 5;
};
class CostModel {
 public:
  explicit CostModel(CostModelOptions options = {});
  // Requires post-propagation/collective IR. At that stage absent internal
  // shardings represent replicated tensors, not an unresolved boundary choice.
  Cost estimate(mlir::ModuleOp module, mlir::sdy::MeshAttr mesh) const;

 private:
  CostModelOptions options_;
};

#endif

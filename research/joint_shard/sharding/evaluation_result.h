#ifndef RESEARCH_JOINT_SHARD_SHARDING_EVALUATION_RESULT_H_
#define RESEARCH_JOINT_SHARD_SHARDING_EVALUATION_RESULT_H_
#include <string>

#include "mlir/IR/Types.h"
#include "research/joint_shard/sharding/boundary_state.h"
#include "research/joint_shard/sharding/cost_model.h"

namespace joint_shard {
struct EvaluationResult {
  bool feasible = false;
  Cost cost;
  std::string failure;
  std::string lowered_mlir;
  BoundaryState boundary;
};
struct ReshardPlan {
  bool feasible = false;
  Cost cost{0, 0, 1};
  TensorSharding from, to;
  mlir::Type type;
  // Empty only for an identity adapter.
  std::string lowered_mlir;
};
}  // namespace joint_shard

#endif

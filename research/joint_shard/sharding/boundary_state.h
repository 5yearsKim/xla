#ifndef RESEARCH_JOINT_SHARD_SHARDING_BOUNDARY_STATE_H_
#define RESEARCH_JOINT_SHARD_SHARDING_BOUNDARY_STATE_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "shardy/dialect/sdy/ir/dialect.h"

namespace joint_shard {
struct Region;

struct MeshContext {
  std::string name;
  mlir::sdy::MeshAttr mesh;
};
// Select an existing mesh; generate data=2,model=2 only for mesh-free inputs.
// Ambiguous multi-mesh inputs require an explicit selection.
MeshContext selectMesh(mlir::ModuleOp module, std::string name = {});

struct TensorSharding {
  mlir::sdy::TensorShardingAttr attr;
  bool operator==(const TensorSharding&) const = default;
  std::string str() const;
};
struct BoundaryState {
  std::vector<TensorSharding> inputs;
  std::vector<TensorSharding> outputs;
  bool operator==(const BoundaryState&) const = default;
  std::string key() const;
};

struct DimensionPolicy {
  std::optional<int64_t> data_dimension = 0;
  std::optional<int64_t> model_dimension = -1;  // Last dimension.
};
struct LayoutPolicy {
  std::string data_axis = "data";
  std::string model_axis = "model";
  DimensionPolicy defaults;
  // Port indices refer to the deterministic interface of each region.
  std::map<size_t, DimensionPolicy> inputs;
  std::map<size_t, DimensionPolicy> outputs;
};

TensorSharding replicatedSharding(mlir::RankedTensorType type,
                                  const MeshContext& mesh);
std::vector<TensorSharding> tensorLayoutChoices(
    mlir::RankedTensorType type, const MeshContext& mesh,
    const LayoutPolicy& policy, const DimensionPolicy& dimensions,
    llvm::ArrayRef<mlir::sdy::TensorShardingAttr> constraints = {});
struct BoundaryEnumeration {
  std::vector<BoundaryState> states;
  bool truncated = false;
};
BoundaryEnumeration enumerateBoundaryStates(const Region& region,
                                            const MeshContext& mesh,
                                            const LayoutPolicy& policy = {},
                                            size_t max_states = 256);
bool validExactSharding(TensorSharding sharding, mlir::RankedTensorType type,
                        const MeshContext& mesh);

}  // namespace joint_shard

#endif

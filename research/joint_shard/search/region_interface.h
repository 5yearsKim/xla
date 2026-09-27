#ifndef RESEARCH_JOINT_SHARD_SEARCH_REGION_INTERFACE_H_
#define RESEARCH_JOINT_SHARD_SEARCH_REGION_INTERFACE_H_
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"

namespace joint_shard {
struct Region;
using ValueId = size_t;
struct TensorPort {
  ValueId value = 0;
  mlir::Type type;
  bool operator==(const TensorPort&) const = default;
};
struct RegionInterface {
  std::vector<TensorPort> inputs, outputs;
};
// IDs are deterministic within one source module: arguments then operation
// results.
class ValueIndex {
 public:
  explicit ValueIndex(mlir::ModuleOp module);
  RegionInterface interface(const Region& region) const;

 private:
  std::map<void*, ValueId> ids_;
};
struct PairInterface {
  RegionInterface external;
  std::vector<size_t> a_inputs;
  // -1 denotes the single A output; other entries index external inputs.
  std::vector<int64_t> b_inputs;
  size_t intermediate_input = 0;
  TensorPort intermediate;
};
PairInterface buildPairInterface(const Region& a, const Region& b,
                                 const ValueIndex& values);
}  // namespace joint_shard

#endif

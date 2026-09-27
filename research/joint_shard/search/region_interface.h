#ifndef RESEARCH_JOINT_SHARD_SEARCH_REGION_INTERFACE_H_
#define RESEARCH_JOINT_SHARD_SEARCH_REGION_INTERFACE_H_
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
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
  TensorPort port(mlir::Value value) const;

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
struct ChainInterface {
  // Function signature order, including unused arguments and repeated returns.
  RegionInterface external;
  std::vector<RegionInterface> regions;
  // -1 is the preceding region's output; otherwise a function argument index.
  std::vector<std::vector<int64_t>> inputs;
  std::vector<size_t> intermediate_inputs;
};
ChainInterface buildChainInterface(mlir::func::FuncOp function,
                                   const std::vector<Region>& regions,
                                   const ValueIndex& values);
struct ValueLifetime {
  TensorPort port;
  // Absent for function arguments. Return uses have index regions.size().
  std::optional<size_t> producer;
  std::vector<size_t> consumers;
};
struct DagInterface {
  RegionInterface external;
  std::vector<RegionInterface> regions;
  std::map<ValueId, ValueLifetime> values;
  // Produced tensors needed by a subsequent region or the function return.
  // Sorted by ValueId. Function arguments are always available separately.
  std::vector<std::vector<TensorPort>> live_after;
};
DagInterface buildDagInterface(mlir::func::FuncOp function,
                               const std::vector<Region>& regions,
                               const ValueIndex& values);
std::vector<std::vector<TensorPort>> buildLiveCuts(
    const RegionInterface& external,
    const std::vector<RegionInterface>& regions);
}  // namespace joint_shard

#endif

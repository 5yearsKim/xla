#ifndef RESEARCH_JOINT_SHARD_TESTS_SUPPORT_FIXTURE_INTERPRETER_H_
#define RESEARCH_JOINT_SHARD_TESTS_SUPPORT_FIXTURE_INTERPRETER_H_
#include <vector>

#include "mlir/IR/BuiltinOps.h"
namespace joint_shard::test {
using DenseTensor = std::vector<float>;
// Deliberately small reference simulator for chain and DAG fixtures.
// Keeps each device's elements in global coordinates and executes communication
// between devices. Supports f32 scalar broadcasts, rank-2 dots, add/multiply,
// tanh and canonical-axis Shardy collectives. Unsupported semantics fail
// loudly.
std::vector<DenseTensor> interpretFixture(
    mlir::ModuleOp module, const std::vector<DenseTensor>& inputs);
}  // namespace joint_shard::test
#endif

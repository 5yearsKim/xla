#ifndef RESEARCH_JOINT_SHARD_TESTS_SUPPORT_REGION_TEST_H_
#define RESEARCH_JOINT_SHARD_TESTS_SUPPORT_REGION_TEST_H_
#include <vector>

#include "research/joint_shard/tests/support/mlir_test.h"
#include "research/joint_shard/transforms/regionizer.h"

namespace joint_shard {
namespace test {
class RegionTest : public MlirTest {
 protected:
  std::vector<Region> regions(mlir::ModuleOp module, size_t max = 128) {
    return Regionizer({max, {}}).split(
        module.lookupSymbol<mlir::func::FuncOp>("main"));
  }
  mlir::OwningOpRef<mlir::ModuleOp> scalingDot() {
    return parse(R"mlir(module {
      func.func @main(%s: tensor<f32>, %a: tensor<8x16xf32>, %b: tensor<16x4xf32>) -> tensor<8x4xf32> {
        %scale = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f32>) -> tensor<8x16xf32>
        %scaled = stablehlo.multiply %scale, %a : tensor<8x16xf32>
        %dot = stablehlo.dot_general %scaled, %b, contracting_dims = [1] x [0]
            : (tensor<8x16xf32>, tensor<16x4xf32>) -> tensor<8x4xf32>
        return %dot : tensor<8x4xf32>
      }
    })mlir");
  }
};
}  // namespace test
}  // namespace joint_shard

#endif

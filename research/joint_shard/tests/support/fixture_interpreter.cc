#include "research/joint_shard/tests/support/fixture_interpreter.h"

#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace joint_shard::test {
namespace {
struct LocalTensor {
  DenseTensor data;
  std::vector<bool> present;
};
using Distributed = std::vector<LocalTensor>;
using Coordinates = std::map<std::string, int64_t>;
mlir::RankedTensorType tensorType(mlir::Value value) {
  auto type = llvm::dyn_cast<mlir::RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape() || !type.getElementType().isF32())
    throw std::invalid_argument(
        "fixture interpreter requires static f32 tensors");
  return type;
}
std::set<std::string> axes(llvm::ArrayRef<mlir::sdy::AxisRefAttr> refs) {
  std::set<std::string> result;
  for (auto axis : refs) {
    if (axis.getSubAxisInfo())
      throw std::invalid_argument(
          "fixture interpreter does not support sub-axes");
    result.insert(axis.getName().str());
  }
  return result;
}
template <typename Lists>
std::set<std::string> flatten(Lists lists) {
  std::set<std::string> result;
  for (auto list : lists) {
    auto names = axes(list);
    result.insert(names.begin(), names.end());
  }
  return result;
}
bool group(const Coordinates& a, const Coordinates& b,
           const std::set<std::string>& varying) {
  for (const auto& [name, coordinate] : a)
    if (!varying.contains(name) && coordinate != b.at(name)) return false;
  return true;
}
bool owns(mlir::Value value, size_t index, const Coordinates& device,
          const std::map<std::string, int64_t>& sizes) {
  auto type = tensorType(value);
  auto sharding = mlir::sdy::getSharding(value);
  if (!sharding) return true;
  for (int64_t dim = type.getRank(); dim-- > 0;) {
    int64_t coordinate = index % type.getDimSize(dim);
    index /= type.getDimSize(dim);
    int64_t shard = 0, count = 1;
    for (auto axis : sharding.getDimSharding(dim).getAxes()) {
      if (axis.getSubAxisInfo())
        throw std::invalid_argument(
            "fixture interpreter does not support sub-axes");
      auto name = axis.getName().str();
      shard = shard * sizes.at(name) + device.at(name);
      count *= sizes.at(name);
    }
    if (type.getDimSize(dim) % count)
      throw std::invalid_argument("uneven fixture sharding");
    if (coordinate / (type.getDimSize(dim) / count) != shard) return false;
  }
  return true;
}
Distributed allocate(mlir::Value value, const std::vector<Coordinates>& devices,
                     const std::map<std::string, int64_t>& sizes) {
  Distributed result;
  size_t count = tensorType(value).getNumElements();
  for (const auto& device : devices) {
    LocalTensor tensor{DenseTensor(count, 0), std::vector<bool>(count)};
    for (size_t i = 0; i < count; ++i)
      tensor.present[i] = owns(value, i, device, sizes);
    result.push_back(std::move(tensor));
  }
  return result;
}
}  // namespace
std::vector<DenseTensor> interpretFixture(
    mlir::ModuleOp module, const std::vector<DenseTensor>& inputs) {
  auto function = module.lookupSymbol<mlir::func::FuncOp>("main");
  if (!function || !function.getBody().hasOneBlock() ||
      function.getNumArguments() != inputs.size())
    throw std::invalid_argument("invalid fixture function");
  std::vector<Coordinates> devices(1);
  std::map<std::string, int64_t> sizes;
  size_t meshes = 0;
  for (auto mesh : module.getOps<mlir::sdy::MeshOp>()) {
    if (++meshes != 1)
      throw std::invalid_argument("fixture interpreter requires one mesh");
    for (auto axis : mesh.getMesh().getAxes()) {
      auto name = axis.getName().str();
      sizes[name] = axis.getSize();
      std::vector<Coordinates> next;
      for (auto device : devices)
        for (int64_t i = 0; i < axis.getSize(); ++i) {
          device[name] = i;
          next.push_back(device);
        }
      devices = std::move(next);
    }
  }
  std::map<void*, Distributed> values;
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto arg = function.getArgument(i);
    auto distributed = allocate(arg, devices, sizes);
    if (inputs[i].size() != distributed[0].data.size())
      throw std::invalid_argument("fixture input shape mismatch");
    for (auto& local : distributed)
      for (size_t j = 0; j < inputs[i].size(); ++j)
        if (local.present[j]) local.data[j] = inputs[i][j];
    values.emplace(arg.getAsOpaquePointer(), std::move(distributed));
  }
  for (auto& op : function.getBody().front().without_terminator()) {
    if (op.getNumResults() != 1)
      throw std::invalid_argument("unsupported fixture operation arity");
    auto output = allocate(op.getResult(0), devices, sizes);
    auto operand = [&](size_t i) -> const Distributed& {
      return values.at(op.getOperand(i).getAsOpaquePointer());
    };
    auto name = op.getName().getStringRef();
    bool reduction = false, communication = false, permute = false;
    std::set<std::string> varying;
    if (auto gather = llvm::dyn_cast<mlir::sdy::AllGatherOp>(&op)) {
      communication = true;
      varying = flatten(gather.getGatheringAxes());
    } else if (auto exchange = llvm::dyn_cast<mlir::sdy::AllToAllOp>(&op)) {
      communication = true;
      for (auto parameter : exchange.getParams()) {
        auto names = axes(parameter.getAxes());
        varying.insert(names.begin(), names.end());
      }
    } else if (auto reduce = llvm::dyn_cast<mlir::sdy::AllReduceOp>(&op)) {
      if (reduce.getReductionOp() != mlir::sdy::ReductionOp::SUM)
        throw std::invalid_argument("only SUM fixture reductions supported");
      communication = reduction = true;
      varying = axes(reduce.getReductionAxes());
    } else if (auto scatter = llvm::dyn_cast<mlir::sdy::ReduceScatterOp>(&op)) {
      if (scatter.getReductionOp() != mlir::sdy::ReductionOp::SUM)
        throw std::invalid_argument("only SUM fixture reductions supported");
      communication = reduction = true;
      varying = flatten(scatter.getReduceScatterAxes());
    } else if (llvm::isa<mlir::sdy::CollectivePermuteOp>(&op)) {
      communication = permute = true;
      auto layout = mlir::sdy::getSharding(op.getOperand(0));
      if (layout && !layout.getUnreducedAxes().empty())
        throw std::invalid_argument(
            "unreduced fixture permutation unsupported");
    }
    if (communication) {
      for (size_t d = 0; d < devices.size(); ++d)
        for (size_t j = 0; j < output[d].data.size(); ++j) {
          if (!output[d].present[j]) continue;
          bool found = false;
          for (size_t source = 0; source < devices.size(); ++source) {
            if (!permute && !group(devices[d], devices[source], varying))
              continue;
            if (!operand(0)[source].present[j]) {
              if (reduction)
                throw std::invalid_argument(
                    "fixture reduction missing contribution");
              continue;
            }
            float x = operand(0)[source].data[j];
            if (reduction)
              output[d].data[j] += x;
            else if (found && output[d].data[j] != x)
              throw std::invalid_argument(
                  "inconsistent fixture collective replicas");
            else
              output[d].data[j] = x;
            found = true;
          }
          if (!found)
            throw std::invalid_argument("fixture collective missing shard");
        }
    } else if (auto dot = llvm::dyn_cast<mlir::stablehlo::DotGeneralOp>(&op)) {
      auto lhs = tensorType(dot.getLhs()), rhs = tensorType(dot.getRhs());
      auto dims = dot.getDotDimensionNumbers();
      if (lhs.getRank() != 2 || rhs.getRank() != 2 ||
          !dims.getLhsBatchingDimensions().empty() ||
          !dims.getRhsBatchingDimensions().empty() ||
          dims.getLhsContractingDimensions().size() != 1 ||
          dims.getLhsContractingDimensions()[0] != 1 ||
          dims.getRhsContractingDimensions().size() != 1 ||
          dims.getRhsContractingDimensions()[0] != 0)
        throw std::invalid_argument(
            "fixture interpreter supports ordinary rank-2 matmul only");
      size_t rows = lhs.getDimSize(0), columns = rhs.getDimSize(1),
             inner = lhs.getDimSize(1);
      for (size_t d = 0; d < devices.size(); ++d)
        for (size_t row = 0; row < rows; ++row)
          for (size_t column = 0; column < columns; ++column) {
            size_t j = row * columns + column;
            if (!output[d].present[j]) continue;
            for (size_t k = 0; k < inner; ++k) {
              size_t a = row * inner + k, b = k * columns + column;
              if (operand(0)[d].present[a] && operand(1)[d].present[b])
                output[d].data[j] +=
                    operand(0)[d].data[a] * operand(1)[d].data[b];
            }
          }
    } else {
      bool scalarBroadcast = false;
      if (auto broadcast =
              llvm::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(&op)) {
        if (tensorType(broadcast.getOperand()).getRank() != 0 ||
            !broadcast.getBroadcastDimensions().empty())
          throw std::invalid_argument(
              "fixture interpreter supports scalar broadcast only");
        scalarBroadcast = true;
      }
      if (!scalarBroadcast && name != "stablehlo.add" &&
          name != "stablehlo.multiply" && name != "stablehlo.tanh" &&
          name != "sdy.all_slice")
        throw std::invalid_argument("unsupported fixture operation: " +
                                    name.str());
      for (size_t d = 0; d < devices.size(); ++d)
        for (size_t j = 0; j < output[d].data.size(); ++j) {
          if (!output[d].present[j]) continue;
          size_t source = scalarBroadcast ? 0 : j;
          if (!operand(0)[d].present[source])
            throw std::invalid_argument(
                "fixture operation missing input shard");
          float x = operand(0)[d].data[source];
          if (name == "stablehlo.add" || name == "stablehlo.multiply") {
            if (!operand(1)[d].present[j])
              throw std::invalid_argument(
                  "fixture binary operation missing shard");
            float y = operand(1)[d].data[j];
            x = name == "stablehlo.add" ? x + y : x * y;
          } else if (name == "stablehlo.tanh")
            x = std::tanh(x);
          output[d].data[j] = x;
        }
    }
    values.emplace(op.getResult(0).getAsOpaquePointer(), std::move(output));
  }
  std::vector<DenseTensor> results;
  auto ret = llvm::cast<mlir::func::ReturnOp>(
      function.getBody().front().getTerminator());
  for (auto value : ret.getOperands()) {
    const auto& distributed = values.at(value.getAsOpaquePointer());
    DenseTensor global(distributed[0].data.size());
    for (size_t j = 0; j < global.size(); ++j) {
      bool found = false;
      for (const auto& local : distributed) {
        if (!local.present[j]) continue;
        if (found && global[j] != local.data[j])
          throw std::invalid_argument("fixture return replicas disagree");
        global[j] = local.data[j];
        found = true;
      }
      if (!found) throw std::invalid_argument("fixture return missing shard");
    }
    results.push_back(std::move(global));
  }
  return results;
}
}  // namespace joint_shard::test

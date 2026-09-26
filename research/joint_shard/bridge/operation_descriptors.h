#ifndef RESEARCH_JOINT_SHARD_BRIDGE_OPERATION_DESCRIPTORS_H_
#define RESEARCH_JOINT_SHARD_BRIDGE_OPERATION_DESCRIPTORS_H_

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"

// Context-owned types/attributes remain valid for the lifetime of the MLIR
// context. Share this table between importer and exporter for one round trip.
class OperationDescriptors {
 public:
  struct Dot {
    mlir::Type lhsType;
    mlir::Type rhsType;
    mlir::Type resultType;
    mlir::DictionaryAttr attributes;
  };

  std::string internDot(Dot descriptor) {
    for (std::size_t i = 0; i < dots_.size(); ++i) {
      const Dot& existing = dots_[i];
      if (existing.lhsType == descriptor.lhsType &&
          existing.rhsType == descriptor.rhsType &&
          existing.resultType == descriptor.resultType &&
          existing.attributes == descriptor.attributes) {
        return "dot_general#" + std::to_string(i);
      }
    }
    dots_.push_back(descriptor);
    return "dot_general#" + std::to_string(dots_.size() - 1);
  }

  const Dot& getDot(const std::string& name) const {
    const std::string prefix = "dot_general#";
    if (name.compare(0, prefix.size(), prefix) != 0) {
      throw std::invalid_argument("invalid dot descriptor name");
    }
    const std::string suffix = name.substr(prefix.size());
    if (suffix.empty() ||
        suffix.find_first_not_of("0123456789") != std::string::npos) {
      throw std::invalid_argument("invalid dot descriptor index");
    }
    return dots_.at(std::stoul(suffix));
  }

 private:
  std::vector<Dot> dots_;
};

#endif  // RESEARCH_JOINT_SHARD_BRIDGE_OPERATION_DESCRIPTORS_H_

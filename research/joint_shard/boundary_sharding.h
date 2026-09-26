#ifndef RESEARCH_JOINT_SHARD_BOUNDARY_SHARDING_H_
#define RESEARCH_JOINT_SHARD_BOUNDARY_SHARDING_H_

#include <string>
#include <vector>

// Each entry describes a closed tensor dimension. Empty means unsharded;
// unused mesh axes are implicitly replicated. Only full mesh axes are
// supported.
struct TensorSharding {
  std::vector<std::vector<std::string>> dimensionAxes;
};

struct BoundarySharding {
  enum class Kind { Argument, Result };
  std::string functionName;
  Kind kind;
  unsigned index;
  TensorSharding sharding;
};

#endif  // RESEARCH_JOINT_SHARD_BOUNDARY_SHARDING_H_

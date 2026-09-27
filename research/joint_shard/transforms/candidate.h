#ifndef RESEARCH_JOINT_SHARD_TRANSFORMS_CANDIDATE_H_
#define RESEARCH_JOINT_SHARD_TRANSFORMS_CANDIDATE_H_
#include <cstddef>
#include <string>
#include <vector>

#include "research/joint_shard/bridge/tensor_lang/tensorlang.h"

namespace joint_shard {
struct Candidate {
  size_t id = 0;
  std::string name;
  TensorRecExpr expression;
  std::vector<size_t> output_roots;
};

}  // namespace joint_shard

#endif

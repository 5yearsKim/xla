#ifndef RESEARCH_JOINT_SHARD_TRANSFORMS_REGION_CANDIDATES_H_
#define RESEARCH_JOINT_SHARD_TRANSFORMS_REGION_CANDIDATES_H_

#include <memory>
#include <string>
#include <vector>

#include "research/joint_shard/transforms/candidate.h"
#include "research/joint_shard/transforms/regionizer.h"
#include "research/joint_shard/transforms/rewrite_options.h"

namespace joint_shard {

using CompiledTensorRules = eggc::CompiledRules<TensorNode, TensorAnalysis>;
CompiledTensorRules compileTensorRules(const TensorRewriteOptions& options);

struct SaturatedRegion {
  std::unique_ptr<TensorEGraph> graph;
  std::vector<eggc::Id> roots;
  Candidate original;
  eggc::RunReport report;
};

SaturatedRegion saturateRegion(const Region& region,
                               const CompiledTensorRules& rules,
                               const TensorRewriteOptions& options);
Candidate mergeExtractedRoots(const std::vector<TensorRecExpr>& roots,
                              std::string name);
// Bounded profile extraction. The cap includes the original; fewer unique
// expressions are expected. Saturation is not repeated for different profiles.
std::vector<Candidate> extractCandidates(const SaturatedRegion& saturated,
                                         const TensorRewriteOptions& options,
                                         size_t max_candidates = 32);

}  // namespace joint_shard

#endif

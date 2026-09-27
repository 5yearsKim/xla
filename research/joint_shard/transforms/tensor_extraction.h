#ifndef RESEARCH_JOINT_SHARD_TRANSFORMS_TENSOR_EXTRACTION_H_
#define RESEARCH_JOINT_SHARD_TRANSFORMS_TENSOR_EXTRACTION_H_

#include <optional>
#include <string>
#include <vector>

#include "eggc/dag_extract.hpp"
#include "research/joint_shard/bridge/tensor_lang/tensor_analysis.h"

namespace joint_shard {

enum class ExtractionProfile { Compute, Depth, Memory };
enum class TensorExtractorMode { Auto, Tree };

struct TensorExtractionReport {
  std::size_t roots = 0;
  bool attempted_dag = false;
  bool selected_dag = false;
  // Describes the DAG search, even if its candidate loses to the baseline.
  bool optimal = false;
  std::optional<eggc::DagStopReason> stop;
  std::size_t explored_states = 0;
  std::size_t peak_frontier = 0;
  std::chrono::nanoseconds search_time{0};
  std::optional<double> baseline_cost;
  std::optional<double> candidate_cost;
  std::string fallback;
};

// Baseline tree extraction is always available. Automatic DAG search applies
// only to a single compute root; a candidate must beat the baseline under the
// same sum of local costs, counting each expression node once.
std::vector<TensorRecExpr> extractTensorRoots(
    const TensorEGraph& graph, const std::vector<eggc::Id>& roots,
    ExtractionProfile profile, TensorExtractorMode mode,
    const eggc::DagOptions& budgets, TensorExtractionReport& report);

}  // namespace joint_shard

#endif  // RESEARCH_JOINT_SHARD_TRANSFORMS_TENSOR_EXTRACTION_H_

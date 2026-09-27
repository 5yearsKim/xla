#ifndef RESEARCH_JOINT_SHARD_REPORTING_ARTIFACT_WRITER_H_
#define RESEARCH_JOINT_SHARD_REPORTING_ARTIFACT_WRITER_H_
#include <string>
#include <utility>

#include "research/joint_shard/search/optimization_observer.h"

namespace joint_shard {
class ArtifactWriter {
 public:
  explicit ArtifactWriter(std::string directory)
      : directory_(std::move(directory)) {}
  // Returned callbacks borrow this writer; keep it alive during optimization.
  OptimizationObserver observer() const;
  void writeSnapshot(const ShardySnapshot& snapshot) const;
  void writeSelectedPrograms(const std::string& selected,
                             const std::string& xla_input) const;

 private:
  void writeText(const std::string& relative, const std::string& text) const;
  std::string directory_;
};
}  // namespace joint_shard

#endif

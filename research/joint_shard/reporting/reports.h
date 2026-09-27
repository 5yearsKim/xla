#ifndef RESEARCH_JOINT_SHARD_REPORTING_REPORTS_H_
#define RESEARCH_JOINT_SHARD_REPORTING_REPORTS_H_
#include <string>

namespace joint_shard {
struct RegionSummary;
struct PairSummary;
struct ChainExperiment;
struct DagExperiment;
struct OptimizationReport;
struct TensorRewriteReport;
std::string formatReport(const RegionSummary& report);
std::string formatReport(const PairSummary& report);
std::string formatReport(const ChainExperiment& report);
std::string formatComparison(const ChainExperiment& report);
std::string formatReport(const DagExperiment& report);
std::string formatComparison(const DagExperiment& report);
std::string formatReport(const OptimizationReport& report);
std::string formatReport(const TensorRewriteReport& report);
}  // namespace joint_shard

#endif

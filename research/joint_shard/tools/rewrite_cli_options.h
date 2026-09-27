#ifndef RESEARCH_JOINT_SHARD_TOOLS_REWRITE_CLI_OPTIONS_H_
#define RESEARCH_JOINT_SHARD_TOOLS_REWRITE_CLI_OPTIONS_H_

#include <string>

#include "cxxopts.hpp"
#include "research/joint_shard/transforms/rewrite_options.h"

void addRewriteCliOptions(cxxopts::Options& options);
bool applyRewriteCliOption(const std::string& name, const std::string& value,
                           TensorRewriteOptions& rewrite_options);

#endif  // RESEARCH_JOINT_SHARD_TOOLS_REWRITE_CLI_OPTIONS_H_

#include "research/joint_shard/tools/rewrite_cli_options.h"

#include <stdexcept>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

namespace joint_shard {
namespace {
TEST(RewriteCliOptionsTest, RewriteCliLimitsAndPolicyAreValidated) {
  TensorRewriteOptions options;
  EXPECT_EQ(options.extractor, TensorExtractorMode::Auto);
  EXPECT_EQ(options.dag.state_limit, 10000);
  EXPECT_EQ(options.dag.time_limit, std::chrono::milliseconds(50));
  EXPECT_EQ(options.dag.frontier_limit, 1000);
  EXPECT_TRUE(applyRewriteCliOption("extractor", "tree", options));
  EXPECT_EQ(options.extractor, TensorExtractorMode::Tree);
  EXPECT_TRUE(applyRewriteCliOption("extractor", "auto", options));
  EXPECT_EQ(options.extractor, TensorExtractorMode::Auto);
  EXPECT_TRUE(applyRewriteCliOption("dag-states", "123", options));
  EXPECT_EQ(options.dag.state_limit, 123);
  EXPECT_TRUE(applyRewriteCliOption("dag-time-ms", "7", options));
  EXPECT_EQ(options.dag.time_limit, std::chrono::milliseconds(7));
  EXPECT_TRUE(applyRewriteCliOption("dag-frontier", "12", options));
  EXPECT_EQ(options.dag.frontier_limit, 12);
  for (const auto& [name, value] :
       std::vector<std::pair<std::string, std::string>>{
           {"extractor", "dag"},
           {"dag-states", "0"},
           {"dag-frontier", "-1"},
           {"dag-time-ms", "bad"},
           {"dag-time-ms", "18446744073709551615"}})
    EXPECT_THROW(applyRewriteCliOption(name, value, options),
                 std::invalid_argument);
  EXPECT_TRUE(applyRewriteCliOption("numerical-policy", "relaxed", options));
  EXPECT_EQ(options.numerical_policy, NumericalPolicy::AllowReassociation);
  EXPECT_TRUE(applyRewriteCliOption("search-visits", "123", options));
  EXPECT_EQ(options.semantic.visit_limit, 123);
  EXPECT_THROW(applyRewriteCliOption("iterations", "0", options),
               std::invalid_argument);
  EXPECT_THROW(applyRewriteCliOption("numerical-policy", "maybe", options),
               std::invalid_argument);
}

TEST(RewriteCliOptionsTest, PermissionOrderingAndExplicitFalseArePreserved) {
  TensorRewriteOptions options;
  EXPECT_TRUE(applyRewriteCliOption("numerical-policy", "strict", options));
  EXPECT_TRUE(applyRewriteCliOption("allow-fp-reorder", "true", options));
  ASSERT_TRUE(options.semantic.permissions);
  EXPECT_TRUE(options.semantic.permissions->reorder_floating_point);
  EXPECT_FALSE(options.semantic.permissions->reassociate_floating_point);
  EXPECT_TRUE(applyRewriteCliOption("numerical-policy", "strict", options));
  EXPECT_FALSE(options.semantic.permissions);
  EXPECT_TRUE(applyRewriteCliOption("allow-fp-reorder", "false", options));
  EXPECT_FALSE(options.semantic.permissions->reorder_floating_point);
  EXPECT_TRUE(applyRewriteCliOption("print-egraph", "false", options));
  EXPECT_FALSE(options.print_egraph);
  EXPECT_FALSE(applyRewriteCliOption("unrecognized", "1", options));
}
TEST(RewriteCliOptionsTest, RegisteredOptionsApplyInCommandLineOrder) {
  cxxopts::Options cli("test");
  addRewriteCliOptions(cli);
  const char* argv[] = {"test", "--numerical-policy=strict",
                        "--allow-fp-reorder", "--dag-states=17",
                        "--print-egraph=false"};
  auto parsed = cli.parse(5, argv);
  TensorRewriteOptions options;
  for (const auto& argument : parsed.arguments())
    ASSERT_TRUE(
        applyRewriteCliOption(argument.key(), argument.value(), options));
  ASSERT_TRUE(options.semantic.permissions);
  EXPECT_TRUE(options.semantic.permissions->reorder_floating_point);
  EXPECT_EQ(options.dag.state_limit, 17);
  EXPECT_FALSE(options.print_egraph);
}
}  // namespace

}  // namespace joint_shard

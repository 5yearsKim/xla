#ifndef RESEARCH_JOINT_SHARD_BRIDGE_TENSOR_LANG_OP_PROPERTIES_H_
#define RESEARCH_JOINT_SHARD_BRIDGE_TENSOR_LANG_OP_PROPERTIES_H_

#include <optional>
#include <span>
#include <string>
#include <variant>

#include "research/joint_shard/bridge/tensor_lang/tensor_analysis.h"

struct Elementwise {
  bool operator==(const Elementwise&) const = default;
};
struct Commutative {
  bool operator==(const Commutative&) const = default;
};
struct Associative {
  bool operator==(const Associative&) const = default;
};
struct Involution {
  bool operator==(const Involution&) const = default;
};
struct LinearIn {
  unsigned operand;
  bool operator==(const LinearIn&) const = default;
};
struct HomogeneousIn {
  unsigned operand;
  bool operator==(const HomogeneousIn&) const = default;
};

using OpProperty = std::variant<Elementwise, Commutative, Associative,
                                Involution, LinearIn, HomogeneousIn>;

enum class NumericalPolicy { PreserveEvaluation, AllowReassociation };

// These are semantic permissions, not a promise of bitwise equivalence.
struct NumericalPermissions {
  bool reorder_floating_point = false;
  bool reassociate_floating_point = false;
  bool distribute_floating_point = false;
  bool assume_finite = false;
  bool ignore_signed_zero = false;
  bool rewrite_dot_arithmetic = false;
};
NumericalPermissions numericalPermissions(NumericalPolicy policy);
struct PropertyDecision {
  bool allowed;
  std::string reason;
};

struct PropertyContext {
  std::span<const TensorFacts> operands;
  TensorFacts result;
  NumericalPolicy policy = NumericalPolicy::PreserveEvaluation;
  std::optional<NumericalPermissions> permissions = std::nullopt;
};

// Candidate properties declared for each operator kind, independent of facts.
std::span<const OpProperty> declaredProperties(OpKind op);

// True only when the operator declares the property and its attrs, facts, and
// numerical policy satisfy that property's requirements.
PropertyDecision queryProperty(const TensorNode& node,
                               const OpProperty& property,
                               const PropertyContext& context);
bool hasProperty(const TensorNode& node, const OpProperty& property,
                 const PropertyContext& context);

#endif  // RESEARCH_JOINT_SHARD_BRIDGE_TENSOR_LANG_OP_PROPERTIES_H_

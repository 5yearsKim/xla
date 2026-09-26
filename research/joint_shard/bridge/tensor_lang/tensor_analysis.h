#ifndef RESEARCH_JOINT_SHARD_BRIDGE_TENSOR_ANALYSIS_H_
#define RESEARCH_JOINT_SHARD_BRIDGE_TENSOR_ANALYSIS_H_

#include <span>
#include <string>

#include "mlir/IR/BuiltinTypes.h"
#include "eggc/analysis.hpp"
#include "eggc/egraph.hpp"
#include "eggc/extract.hpp"
#include "research/joint_shard/bridge/tensor_lang/tensorlang.h"

// Semantic value facts only. Candidate layouts, cost, and intrinsic operator
// properties do not belong here. A null type means unknown analysis
// information, distinct from a ranked tensor type containing dynamic
// dimensions.
struct TensorFacts {
  mlir::RankedTensorType type;
  // A known constant value, when all equivalent nodes agree.
  mlir::ElementsAttr constant = {};
};

enum class InferenceStatus { Valid, Unknown, Invalid };
struct InferenceResult {
  InferenceStatus status;
  TensorFacts facts;
  std::string reason;
  bool valid() const { return status == InferenceStatus::Valid; }
};
// Does not touch the graph; suitable for validating an entire proposed RHS.
InferenceResult inferTensorNode(const TensorNode& node,
                                std::span<const TensorFacts> operands);
bool canonicalReductionIdentity(const ReduceAttrs& attrs,
                                mlir::Type element_type);

struct TensorAnalysis;
using TensorEGraph = eggc::EGraph<TensorNode, TensorAnalysis>;
struct TensorAnalysis {
  using Data = TensorFacts;
  Data make(const TensorEGraph& graph, const TensorNode& node) const;
  eggc::AnalysisMerge merge(Data& into, const Data& from) const;
};
using TensorExtractor = eggc::Extractor<TensorNode, TensorAnalysis>;

const TensorFacts& tensorFacts(const TensorEGraph& graph, eggc::Id id);

#endif  // RESEARCH_JOINT_SHARD_BRIDGE_TENSOR_ANALYSIS_H_

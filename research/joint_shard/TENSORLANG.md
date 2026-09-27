# TensorLang implementation and extension guide

The implementation keeps TensorLang semantics in `bridge/tensor_lang/` and the
MLIR bridge, region driver, and Shardy pipeline in their existing directories.
No descriptor table, operator-string encoding, or TensorLang dependency in
`egg-c` is needed.

| File | Responsibility |
| --- | --- |
| `tensorlang.h/.cc` | Native nodes, typed semantic attrs, shared operator schema, equality/hash |
| `tensor_analysis.h/.cc` | Standalone inference and monotone e-class type/constant facts |
| `op_properties.h/.cc` | Candidate properties by OpKind and contextual decisions with rejection reasons |
| `semantic_rewrite.cc` | DSL parser, validation, bounded streaming matcher, complete RHS preparation |
| `tensor.rules` | Default declarative rules, embedded into the library during Bazel builds |
| `attribute_rewrites.cc` | Computed-attribute shape and dot rewrites |
| `tensor_rewrites.h/.cc` | Public rule APIs and a small ordinary-pattern example |
| `../stablehlo_importer.cc`, `../stablehlo_exporter.cc` | Conservative MLIR admission and typed round trips |
| `../../transforms/rewrite_regions.cc` | Multi-output region sessions, saturation, shared extraction/export |
| `../../sharding/shardy_runner.cc` | Shardy snapshots and candidate cost estimates |

## Node and inference contracts

```cpp
TensorNode node{OpKind::Transpose, TransposeAttrs{{0, 2, 1}}, {x}};
```

`op`, `attrs`, and operands determine identity. `matches` compares kind, attrs,
and arity while ignoring operand IDs. MLIR handles in attrs are immutable and
context-owned; retain the context throughout graph use and export.

`opSchema` / `lookupOpSchema` provide names, arities, and whether a concrete DSL
operator can use `NoAttrs`. Add new operations here and implement attribute
validation, inference, importer, and exporter together.

`inferTensorNode(node, operandFacts)` returns **Valid**, **Unknown**, or
**Invalid**, with an explanation. It does not access or mutate an e-graph.
Unknown facts reject a speculative RHS. TensorAnalysis uses the same inference
and rejects conflicts before a node/union is inserted. Constant facts refine
from unknown to a proven value; incompatible known types or values conflict.
E-class facts do not store root-operator properties or optimization costs.

The exporter validates the complete extracted DAG before creating operations.
`exportRoots` shares its value cache across outputs. Input indices refer to the
original region's ordered boundary-value vector.

## Operator properties and numerical semantics

`declaredProperties` defines candidate properties directly inside the OpKind
switch. `queryProperty` checks attrs, facts, shape, and numerical permissions and
returns `PropertyDecision{allowed, reason}`. `hasProperty` is its boolean wrapper.

| Operator | Candidate properties |
| --- | --- |
| Add, Maximum, Minimum | Elementwise, Commutative, Associative |
| Multiply | Elementwise, Commutative, Associative, LinearIn(0/1), HomogeneousIn(0/1) |
| DotGeneral | LinearIn(0/1), HomogeneousIn(0/1) |
| Negate | Elementwise, Involution, LinearIn(0), HomogeneousIn(0) |
| Transpose | Involution, LinearIn(0), HomogeneousIn(0) |
| Reshape, BroadcastInDim | LinearIn(0), HomogeneousIn(0) |
| Reduce | LinearIn(0), HomogeneousIn(0), restricted to canonical sum |
| Other enabled elementwise ops | Elementwise |

These are conditional contracts. Transpose is an involution only for a
self-inverse permutation. Data-moving linearity requires static shapes.
Integer arithmetic uses its modular algebra. Floating and complex arithmetic
are conservative in the default **strict** policy: no operand swapping,
reassociation, distribution, or arithmetic involution is enabled. Layout
composition and validated data movement remain available.

**Relaxed** (`AllowReassociation`) explicitly permits floating operand
reordering, reassociation, distribution, and dot arithmetic under finite-value
and signed-zero assumptions. It does not establish bitwise equality, bound
roundoff, or prove that runtime values satisfy those assumptions. Dot algorithm
attrs and unknown extra attrs reject algebraic dot rules even in relaxed mode;
precision configuration is retained unchanged. Quantized/mixed-dtype/encoded
dots are not supported by the current inference contract.

For narrower permissions, set `PropertyContext::permissions` or
`SemanticRuleOptions::permissions`. The CLI accepts `--allow-fp-reorder`,
`--allow-fp-reassociate`, `--allow-fp-distribute`, `--assume-finite`,
`--ignore-signed-zero`, and `--allow-dot-arithmetic`. A numerical-policy option
resets overrides; flags following it refine that preset.

## Semantic DSL

```text
rule factor-right-linear-operator-from-add {
  add(F(?a, ?x), F(?a, ?y)) => F(?a, add(?x, ?y))
  where LinearIn(F, 1);
}

rule commute-elementwise {
  F(?x, ?y) => F(?y, ?x)
  where Elementwise(F) and Commutative(F);
}
```

Tensor variables begin with `?`. Operator variables start uppercase. Concrete
names resolve through the schema and currently require attribute-free operators;
attribute-bearing operators are captured as variables or handled in C++ rules.
Load validation checks rule-name uniqueness, concrete names/arity, consistent
operator-variable arity, bound RHS variables/operators, and predicate indices.
Diagnostics include the source path and line/column; semantic validation points
to the rule's declaration line.

Operator variables capture kind, complete attrs, and arity. Repeated occurrences
must match that identity; their operands and result facts are checked separately.
The matcher streams depth-first matches, uses discriminant indexes where
possible, deduplicates bindings, checks cancellation inside recursion, and limits
node visits and structural matches, including rejected matches.

Every property is checked on every captured occurrence and on RHS operator
occurrences. The complete RHS is inferred and its root type compared with the
matched root before emission. It is revalidated before graph insertion. A bad
parent cannot leave speculative child nodes in the graph. Rules are trusted
algebraic equations supplied by the author: the parser validates their structure
and applicability, not an arbitrary equation's mathematical truth.

Value predicates inspect explicitly named tensor variables: `Scalar(?s)`
requires a known rank-zero type; `Uniform(?v)` accepts a scalar, a known dense
splat, or broadcasts of a proven uniform value. These checks share the rule's
cancellation and visit budgets. `HomogeneousIn` checks the operator's law and
numerical permissions; it no longer implicitly looks for a variable named
`?scale`. Authors must express their value requirements in the equation/guards.

The DSL helper `scale(?s, ?x)` represents multiplication by a rank-zero scalar:

```text
rule move-scaling-out-of-left-operand {
  F(scale(?s, ?x), ?y) => scale(?s, F(?x, ?y))
  where HomogeneousIn(F, 0) and Scalar(?s);
}
```

On the LHS, it matches a multiply and follows broadcast chains to bind the
original rank-zero scalar. Matching the scalar as the right operand also checks
multiply commutation permissions, since the RHS puts it first; strict floating
evaluation therefore requires the scalar on the left. On the RHS, it prepares a
`BroadcastInDim` with empty dimensions and the inferred tensor result type,
then a `Multiply`; rank-zero tensors need no broadcast. Thus a scale broadcast
to `2x3` before a transpose becomes a broadcast to `3x2` after it, and a scale
on either input of a `2x3` by `3x4` dot becomes a broadcast to `2x4`. Scalar and
tensor element types must agree. The helper adds no new TensorLang operation
and requires no egg-c changes. The full constructed RHS is validated before
any nodes are inserted.

A tensor splat without rank-zero broadcast provenance does not match `scale`.
Use `Uniform(?v)` in an ordinary multiply rule if its scale already has the
correct RHS shape. Dynamic output construction is limited to what standalone
broadcast inference can prove; homogeneity still requires the operator's
existing shape and numerical guards.

`defaultSemanticRules()` returns the embedded `tensor.rules`; execution does not
rely on the current directory. `--rules=path` replaces this default DSL rule set.
Paths are caller-resolved. C++ attribute rules are always appended. Rule groups
are `exact` (commutation/involution), `algebra` (also associativity), and `all`
(also linearity/homogeneity); numerical guards apply in every group.

The default driver loads rules once before mutation. It allows 10 iterations,
10,000 nodes, 4,096 structural matches per iteration, initial per-rule backoff at
256 matches, and a cooperative 1-second limit per region. Semantic searches also
limit visits to 100,000 and distinct structural matches to 10,000 per rule/search.
Attribute searches use a fixed 100,000-visit cap. Interrupted search iterations
discard pending applications. `SearchLimit` distinguishes custom budget
exhaustion from `TimeLimit`. Rebuild/application time limits are cooperative.

`TensorRewriteReport` records import coverage/boundaries, per-rule visits,
structural matches, duplicates, property/RHS rejection reasons, accepted/applied
counts, budget stops, and engine run history. Semantic rules emit both accepted and rejected structural matches, so the
engine's global/per-rule budgets and condition statistics include rejections.
The semantic report additionally records the reason for each rejection.

## Bridge coverage and computed-attribute rules

Import/export cover add, subtract, multiply, divide, maximum, minimum, negate,
exp, log, sqrt, tanh, constants, transpose, static reshape, broadcast_in_dim,
and general/batched dot_general with same element types. Canonical single-input
sum/product/max/min reductions preserve a scalar identity initializer and their
axes. Import validates the reducer body, argument/result types, and initializer;
export reconstructs the body. Arbitrary, multi-input, nonidentity, and complex
reductions remain boundaries. Empty reductions preserve the initializer.

Unannotated `sdy.constant` is normalized to `stablehlo.constant`. Explicit
shardings, function attrs, constraints (including dangling constraints), unknown
metadata, and unsupported operations are preserved in original MLIR.
`result_layout` and `xla_shape` are classified as derived lowering hints and
omitted on reconstruction; downstream lowering chooses/recomputes them. All
other unknown attrs are boundaries. Dot dimensions, precision, algorithm, and
result type are semantic attrs; managed dot attrs cannot also occur in the extra
attribute dictionary.

C++ rules eliminate identity layout operations, compose arbitrary transpose
permutations, collapse reshapes, compose broadcast dimension maps, and absorb
an input transpose into dot dimension lists. Dot absorption requires free axes
to retain their result ordering; transformations needing an output transpose
are deliberately outside this rule. Proposed nodes use standalone inference
before insertion.

Each supported contiguous island uses one e-graph for all values consumed
outside the island. An extractor is shared across roots; its selected expression
DAGs are deduplicated and exported with one cache. Unsupported/annotated ops
separate islands, so this is not whole-function optimization across constraints.
Compute, depth, and memory extraction profiles replace the old add-order demo.
The default `--extractor=auto` uses bounded egg-c DAG extraction for islands
with one output and the compute profile. Shared computations are charged once.
The existing tree extractor supplies a baseline; both expressions are scored
by summing the same local operator weights once per expression node. A DAG
candidate replaces the baseline only on a strict cost improvement, including
when search stops with a finite candidate before proving optimality. Ties,
unavailable candidates, multiple outputs, and depth/memory profiles retain tree
extraction. `--extractor=tree` disables DAG search for comparison.

Each eligible island has independent defaults of `--dag-states=10000`,
`--dag-time-ms=50`, and `--dag-frontier=1000`. Limits cover search, not baseline
extraction or export. Time checks are cooperative, and the frontier budget
counts retained frames rather than bytes. `--rewrite-report` records the
selected extractor, fallback reason, baseline/candidate compute costs, search
time, explored states, peak frontier, stop reason, and whether the DAG search
proved optimality. An optimal search does not imply measured runtime optimality.

The memory profile uses logical intermediate element counts, not peak liveness.
It and the depth profile keep their existing tree objectives; no multi-output
DAG or scheduling/liveness optimizer is implemented.

## Candidate selection and next extensions

`run_shardy --round-trip` evaluates the original plus one rewritten candidate.
`--candidates=1..4` includes the original plus bounded compute/depth/memory
candidates. Each runs on an independent clone through Shardy; a failed rewritten
candidate cannot displace the baseline. Comparisons require the collective stage.
The score is estimated logical collective payload bytes, then compute work.
Ties preserve the original. Unknown/overflowing estimates do not displace the
baseline. Dumps are separated by candidate. Stdout is the selected verified
module; diagnostics and rewrite reports go to stderr.

The payload metric sums the largest global operand/result tensor payload per
collective. It is a topology-independent proxy, **not** actual network bytes or
latency: device groups, local partitions, and collective algorithms are not yet
modeled. Compute work estimates dot multiply/adds, elementwise/reduction work, and
transpose element movement. Selection is among a few scalar extraction profiles, not a k-best/layout
DP or automatic search over new boundary shardings.

Useful next extensions are per-device/topology-aware communication costs,
layout-conditioned extraction, uniform-scalar facts beyond simple provenance,
broader dtype/algorithm contracts, dynamic shape proof
facts, and numerical equivalence gates for additional operations and floating policies. Keep those
proof facts separate from estimated candidate costs.

## Validation and third-party maintenance

Tests cover DSL validation, source diagnostics, conjunctions, numerical policy,
invalid-RHS nonmutation, search limits, dot factorization (including an independent
modular integer numerical oracle), rectangular transpose/dot scalar scaling,
explicit scalar/uniform guards and rejection without graph mutation,
non-involutive transpose composition, shared outputs, batched dots, reductions,
empty reductions, metadata boundaries, dot transpose absorption, costs, and CLI
options. Extraction tests additionally cover a modular integer dot equation
where preserving sharing beats tree factorization, state/time/frontier budgets,
worse and improved partial candidates, ties, and depth/memory/multiple-output
fallbacks. Pipeline tests check shared export and DAG diagnostics. Existing
fixed-boundary tests expect strict evaluation order.

Run these commands from the XLA workspace root:

```sh
bazel build --config=joint_shard //research/joint_shard/tools:parse_stablehlo //research/joint_shard/tools:run_shardy
bazel test --config=joint_shard //research/joint_shard:tests --test_output=errors
bazel-bin/research/joint_shard/tools/parse_stablehlo research/joint_shard/testdata/megatron_layer/01.before_propagation.mlir --rewrite-report
bazel-bin/research/joint_shard/tools/run_shardy research/joint_shard/testdata/dot_gather.mlir --candidates=4 --rewrite-report --dump-dir=/tmp/joint_shard
```

Use `--numerical-policy=relaxed` explicitly to explore floating algebraic
candidates. `--iterations`, `--nodes`, `--matches`, `--search-visits`, `--time-ms`,
`--extraction`, `--extractor`, `--dag-states`, `--dag-time-ms`, and
`--dag-frontier` customize the driver. C++20 and the existing
`--config=joint_shard` toolchain configuration are required.

The additional generic engine changes are `StopReason::SearchLimit` for a
cancelled custom search without a time/global-match limit, and an optional
`Application::condition_result` so custom searchers can count rejected matches
without an applier. They carry no TensorLang dependency. Maintain engine edits in an egg-c fork or
upstream commit, then update the parent submodule pin. No commits or dependency
pin changes were made here. The existing engine architecture and its Language /
AnalysisFor contracts are described in [egg-c/docs/custom_languages.md](egg-c/docs/custom_languages.md).

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
| `tensor_patterns.h/.cc` | Typed C++ pattern builders, guards and checked callback expressions |
| `semantic_rewrite.cc` | Pattern compilation and rule-group filtering |
| `semantic_rule_validation.cc` | Variable bindings, arities and rule definition validation |
| `semantic_rule_matcher.cc` | Bounded streaming tensor/operator/attribute matching and value predicates |
| `semantic_rule_attributes.cc` | Typed field contracts, concrete attribute matching and checked RHS construction |
| `semantic_rule_guards.cc` | Numerical guards for dedicated arithmetic rules |
| `semantic_rule_application.cc` | Property checks, complete RHS preparation and rule application |
| `semantic_rule_internal.h` | Private AST, binding and search contracts |
| `tensor_rules.cc` | Default C++ definitions and dot signature table |
| `attribute_rewrites.cc` | Computed-attribute shape and dot rewrites |
| `tensor_rewrites.h` | Public semantic/attribute rule APIs and diagnostic types |
| `../stablehlo_importer.cc`, `../stablehlo_exporter.cc` | Conservative MLIR admission and typed round trips |
| `../../transforms/rewrite_regions.cc` | Multi-output region sessions, saturation, shared extraction/export |
| `../../sharding/shardy_runner.cc` | Shardy stage execution and optional snapshots |
| `../../sharding/cost_model.cc` | Per-device optimizer time estimates |
| `../../sharding/module_statistics.cc` | Optional global work/payload diagnostics |

All C++ APIs use the `joint_shard` namespace.

## Node and inference contracts

```cpp
joint_shard::TensorNode node{joint_shard::OpKind::Transpose,
                            joint_shard::TransposeAttrs{{0, 2, 1}}, {x}};
```

`op`, `attrs`, and operands determine identity. `matches` compares kind, attrs,
and arity while ignoring operand IDs. MLIR handles in attrs are immutable and
context-owned; retain the context throughout graph use and export.

`opSchemas` / `opSchema` / `lookupOpSchema` provide canonical StableHLO names, arities, and whether
a concrete pattern operator can use `NoAttrs`. Add new operations here and implement attribute
validation, inference, importer, and exporter together. Property-indexed semantic
search iterates registered schemas rather than relying on enum ordering.
`stableHloName` uses that same table. Short text-DSL operator-name aliases are
no longer accepted; concrete patterns use `OpKind`.

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
returns `PropertyDecision{allowed, reason}`.

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
are conservative in the **strict** policy: no operand swapping,
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
`--ignore-signed-zero`, `--allow-dot-arithmetic`, and `--allow-dot-division`.
Moving division across a dot additionally assumes nonzero finite denominators
and tolerable changes in intermediate range and rounding. Relaxed mode enables
this permission; `--numerical-policy=relaxed --allow-dot-division=false` disables
it independently. The engine does not prove these runtime assumptions.
A numerical-policy option
resets overrides; flags following it refine that preset.

`--allow-raw-moments` is a separate aggressive permission, disabled in **every**
preset, including relaxed. Use
`--numerical-policy=relaxed --allow-raw-moments` to enable centered-square
reductions to raw moments. This deliberately accepts cancellation and possible
negative variance; it does not clamp or stabilize the replacement. The other
floating reassociation/distribution, finite-value, and signed-zero permissions
must also be enabled.

## C++ pattern definitions

Include `bridge/tensor_lang/tensor_patterns.h` and use the
`joint_shard::patterns` namespace. Tensor, operator, and attribute variables are
explicit C++ types; capitalization has no meaning. Concrete operation builders
use `OpKind`, and operator variables capture kind, complete attrs, and arity.

```cpp
using namespace joint_shard;
using namespace joint_shard::patterns;
const auto a = tensor_var("a"), x = tensor_var("x"), y = tensor_var("y");
const auto F = operator_var("F", 2);
auto definition = rule("factor-right", add(F(a, x), F(a, y)),
                       F(a, add(x, y)))
                      .when(linear_in(F, 1));
auto rewrites = compileRules({definition}, NumericalPolicy::AllowReassociation);
```

`rule(name, lhs, rhs).when(...)` describes an equation. `.when` accepts one
guard or an initializer list of guards. Copying a definition and adding guards
leaves the original unchanged. Concrete attribute-free operations have ordinary
C++ builders such as `add`, `multiply`, and `negate`; `operation(OpKind, operands)`
provides access to the other attribute-free operators.

Attributes are typed literals or `attribute_var<T>` bindings. Dot dimensions
are captured together as `DotDimensions`, which contains the contracting and
batching lists for both operands. `dot_dims` constructs a literal. Repeated
variables must match exactly; variables used in the RHS must be bound on the
LHS. A name cannot be reused across tensor, operator, or attribute types.

```cpp
const auto x = tensor_var("x"), w = tensor_var("w");
const auto p = attribute_var<mlir::ArrayAttr>("precision");
auto project_first = rule(
    "project-before-sequence-sum",
    dot_general(reduce(x, ReduceKind::Sum, Axes{1}), w, dot_dims({1}, {0}), p),
    reduce(dot_general(x, w, dot_dims({2}, {0}), p), ReduceKind::Sum, Axes{1}))
    .when({rank(x, 3), rank(w, 2), sum_dot_interchange()});
```

| Constructor | Typed attributes | Contract |
| --- | --- | --- |
| `dot_general(a, b, dims, precision)` | `DotDimensions`, `mlir::ArrayAttr` | Omitted precision matches absence; a variable captures the exact configuration including absence. Algorithms and nonempty extra attrs do not match. RHS result type is inferred. |
| `reduce(x, kind, axes)` | `ReduceKind`, `Axes` | Only canonical identity initializers match. The RHS identity is constructed for the operand dtype. |
| `transpose(x, permutation)` | `Axes` | Result type follows the operand and permutation. |
| `broadcast_in_dim(x, attrs)` | `BroadcastAttrs` | Captures both dimensions and destination type. Copying attrs preserves that type; use a callback to construct a broadcast at another shape. |

C++ checks constructor types and guard signatures. Constructors check operator
arity. `compileRules` checks variable bindings, consistent attribute types and
operator arities, unique rule names, guard subjects, and that each definition
has exactly one RHS. It validates even definitions disabled by a rule group.
Errors identify the rule name. The equations remain trusted: type and numerical
checks do not prove the mathematical truth of a custom equation.

### Computed attributes and custom replacements

Use `rule(name, lhs).build(callback)` when replacement attributes depend on the
match. A callback receives a read-only `Match` and an `RhsBuilder`. `match[x]`
returns the tensor's e-class ID; `match.type(x)` returns its inferred type;
`match[attr]` returns a typed attribute. Compute dimensions or permutations in
ordinary C++, then return a checked expression. No additional language or
attribute-function registry is needed.

`match.constant(x)` returns the proven `mlir::ElementsAttr`, or a null attribute
when no constant is known. Callbacks can use this to validate semantic constants
such as the divisor in a mean; a tensor variable alone does not prove its value.

For example, composing arbitrary valid permutations needs a small callback:

```cpp
const auto x = tensor_var("x");
const auto p = attribute_var<Axes>("inner"), q = attribute_var<Axes>("outer");
auto compose = rule("compose-transposes", transpose(transpose(x, p), q))
    .build([=](const Match& match, RhsBuilder& rhs) {
      auto inner = match[p], outer = match[q];
      if (inner.size() != outer.size()) return rhs.reject("ranks differ");
      Axes result;
      for (auto axis : outer) result.push_back(inner.at(axis));
      return rhs.transpose(rhs.ref(match[x]), std::move(result));
    });
```

`RhsBuilder` provides checked `dot_general`, `reduce`, `transpose`,
`broadcast_in_dim`, `divide`, `negate`, and a general
`operation(OpKind, attrs, operands)` constructor.
`apply(F, operands)` uses a captured operator's attrs and preserves its property
checks at the new operand shapes. `reject(reason)` declines a structural match.
Unbound or wrongly typed callback bindings become reported rejections. Other
C++ exceptions propagate as programming errors. Keep captures owned by value;
callbacks may run repeatedly, should be pure, and must keep any referenced MLIR
context alive. The matcher can stop between callbacks; custom computation must
be bounded to retain cooperative time limits.

Builder operations prepare temporary expressions and never add nodes to the
e-graph. The compiler validates the entire tree and root type before emitting
an application, and rechecks it before insertion. Failed parents and explicit
rejections leave no speculative child nodes. This same path handles declarative
and callback replacements. Numerical guards apply to both.

### Guards and default rules

`rank(x, n)`, `scalar(x)`, and `uniform(x)` check known tensor facts. `uniform`
accepts a scalar, a proven dense splat, or broadcasts of a proven uniform value.
`commutative(F)`, `associative(F)`, `involution(F)`, `linear_in(F, i)`,
`homogeneous_in(F, i)`, and `elementwise(F)` use the operator property contracts.
Every captured occurrence and every rebuilt occurrence is checked separately.
Repeated operator variables must agree on kind, attrs, and arity.

`dot_reassociation()`, `sum_dot_interchange()`, and `dot_arithmetic()` are numerical applicability
guards rather than generic properties. They inspect explicit matched operations
and prepared replacements, without traversing opaque tensor-variable subgraphs.
They require static types and real floating or modular integer arithmetic.
Floating use requires dot-arithmetic, reassociation, distribution, finite-value,
and signed-zero permissions. Only absent, empty, or paired typed StableHLO
DEFAULT precision entries are admitted, and captured precision is retained.
Nondefault precision, algorithms, complex arithmetic, and unknown dot metadata
are rejected. Generic `associative(F)` never admits DotGeneral.

`dot_division()` adds the dot-division permission and rejects integer division.
The default `dot-divide-broadcast` rules support static matrix and batch-matrix
products. Direct and two-stage broadcasts are matched explicitly, the denominator
must be invariant along the contracting axis, and a singleton mapped onto that
axis is dropped with a reshape. A checked callback builds the output broadcast
at the dot result shape. The rule has one direction to limit search growth.

`feature-gram-to-sample-gram` rewrites the squared Frobenius norm of a rank-2
cross-product into the inner product of two sample Gram matrices. Explicit
transposes and folded dot dimensions are supported. The matrices need a shared
sample count, but their feature counts can differ. Real floating use requires
`dot_arithmetic()` permissions; modular integers are admitted in strict mode.
This rule covers the isolated norm, not Barlow's normalization or diagonal gather.

`raw_moments()` guards two centered-square reduction patterns, with direct and
JAX keepdims broadcasts:
`sum((x - sum(x)/N)^2) -> sum(x*x) - sum(x)*sum(x)/N`.
The callback checks canonical sum reductions, a static positive reduction size,
a proven scalar floating constant equal to that size, and the complete mean
broadcast mappings. Only real floating tensors are admitted. Integer division
and arbitrary mean divisors are rejected. The original outer division by N,
epsilon, rsqrt, affine parameters, and projection stay in the expression; their
existing rules can subsequently compose with this one. No LayerNorm node or
collective operator is added to the pattern language.

`scale(s, x)` matches multiplication by a rank-zero scalar through broadcast
chains. Matching the scalar on the right also requires multiply commutation
permission. On the RHS it constructs a broadcast at the new tensor shape;
rank-zero tensors need no broadcast. The scalar and tensor dtype must agree.
A tensor splat without scalar broadcast provenance does not match this helper.

`tensor_rules.cc` is the default rule source. `buildSemanticRules(policy, options)`
compiles those definitions, and the driver appends the existing computed shape
and dot rewrites from `buildAttributeRewrites()`. Text rule files, their parser,
embedded rule generation, and `--rules` have been removed. Add or change rules
in C++ and rebuild.

The defaults retain the generic property rules and forward/reverse matrix,
batched, batched/shared-weight, shared-weight, and kernel-attention dot chains.
Compact signature tables construct dot-chain and sum/dot variants. Row, sequence,
spatial, and weight sums plus kernel denominators have both directions. Direct
LoRA rules and coordinated kernel numerator/denominator rules avoid requiring
several intermediate rewrites to discover the full equation. See
[WORKLOAD_REWRITES.md](WORKLOAD_REWRITES.md) for the workload checks and omitted
cases. There is no general dimension-lineage solver; custom generalizations can
use `.build()`.

Rule groups are `exact` (commutation/involution), `algebra` (also associativity
and dedicated dot reassociation), and `all` (also linearity/homogeneity and
sum/dot interchange). Numerical guards remain active in every group. Matching
streams depth-first, uses discriminant indexes where possible, deduplicates
bindings, and shares cancellation, visit, and structural-match budgets.

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

`summarize_regions` owns joint region candidate/layout search. It saturates each
region once, preserves the original source expression, extracts bounded unique
profile candidates with ordered shared roots, evaluates each under exact boundary
states on independent Shardy clones, and preserves exact boundary tables alongside a reshard-pruned region frontier.
Optional `--compose-regions=A,B` searches adjacent region pairs and materializes
verified child-plus-adapter modules for each external boundary.
Composition automatically adds executable adapters for dominance replacements.
Driver options default to relaxed floating-point algebra. Low-level semantic
APIs retain explicit strict/relaxed policies and all existing guards.

`--max-candidates=32` is a cap, not a guarantee of 32 candidates. This milestone
uses original, compute tree/DAG, depth and memory alternatives. It does not yet
have k-best or sharding-aware extraction. `run_shardy` now only runs the annotated
pipeline; its previous candidate comparison and global lexicographic selection
were removed. Global payload/work counters are diagnostics. The optimizer uses
one additive per-device work/collective cost model for execution and adapters.
See [REGION_OPTIMIZER.md](REGION_OPTIMIZER.md) for architecture, policies, tests,
limits and experiments.

Useful next extensions are per-device/topology-aware communication costs,
layout-conditioned extraction, uniform-scalar facts beyond simple provenance,
broader dtype/algorithm contracts, dynamic shape proof
facts, and numerical equivalence gates for additional operations and floating policies. Keep those
proof facts separate from estimated candidate costs.

## Validation and third-party maintenance

Tests cover typed attribute matching/construction, variable conflicts, computed
callback replacements and rejection, exact precision preservation and rejection, every floating guard
permission, invalid-RHS nonmutation, rule-group filtering and cancellation.
An independent dense interpreter checks rectangular and batched matrix chains,
sequence sum/projection and kernel denominators in both directions against
modular integer and floating references. Pipeline tests verify selected RHS
shapes and exported precision/reducer bodies. Tests also cover C++ binding validation,
rule-name diagnostics, conjunctions, numerical policy,
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
bazel build --config=joint_shard //research/joint_shard/tools:parse_stablehlo //research/joint_shard/tools:run_shardy //research/joint_shard/tools:summarize_regions
bazel test --config=joint_shard //research/joint_shard:tests --test_output=errors
bazel-bin/research/joint_shard/tools/parse_stablehlo research/joint_shard/testdata/megatron_layer/01.before_propagation.mlir --rewrite-report
bazel-bin/research/joint_shard/tools/summarize_regions research/joint_shard/testdata/scaling_dot.mlir --dump-dir=/tmp/joint_shard_regions
```

Relaxed floating algebra is the driver default. Use
`--numerical-policy=strict` for strict evaluation. `--iterations`, `--nodes`, `--matches`, `--search-visits`, `--time-ms`,
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

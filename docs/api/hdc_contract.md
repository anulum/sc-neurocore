# HDC/VSA Semantic Contract

Reference-locked semantics for the `sc_neurocore.hdc` package. Every
rule below is enforced by an executed test named in the final section;
a behavioural change to any rule is a breaking contract change and
must update this page and its tests in the same commit.

## Representation

- Hypervectors are **binary {0, 1}** NumPy arrays of dtype `uint8` and
  shape `(dim,)`.
- `dim` is a positive integer; construction rejects anything else with
  a typed `ValueError`. D ≥ 10,000 is the recommended operating regime
  for quasi-orthogonality; the algebra itself is exact at any dim.
- The bipolar view used by the centroid classifier is `2·v − 1` over
  `int64` accumulators; the binary form remains canonical at the API
  boundary.

## Deterministic generator and seed contract

- All randomness flows through one `numpy.random.Generator`
  (`default_rng(seed)`) owned by the encoder. No operation ever draws
  from NumPy's process-global stream.
- A seeded encoder is fully deterministic **given the same call
  order**: the n-th draw after construction is identical across
  processes and platforms supported by NumPy's PCG64 stream.
- `item(name)` draws on first request and caches; the mapping
  name → vector is therefore deterministic per (seed, first-request
  order) and stable for the lifetime of the encoder. `item` returns a
  defensive copy.
- The `"random"` bundle tie policy consumes one draw from the same
  generator per tied bundle call — and none when no position ties.

## Operations

| Operation | Definition | Contract |
|-----------|------------|----------|
| Bind | `XOR(v1, v2)` | self-inverse: `a⊗a = 0`, `(a⊗b)⊗b = a`; commutative; distance-preserving |
| Bundle | position-wise majority | strict majority of ones → 1, of zeros → 0; exact ties (even counts only) follow `tie_policy` |
| Permute | `numpy.roll(v, shifts)` | cyclic **right** rotation for positive `shifts`: `ρ(v)[i] = v[(i − shifts) mod D]`; `ρ_a∘ρ_b = ρ_{a+b}`; `ρ_{-k}` inverts `ρ_k` |

### Bundle tie policy

Exactly tied positions can occur only for even vector counts.

- `"zeros"` (default): tied positions become 0 — the historical
  strict-majority behaviour. This is a documented deterministic bias
  toward 0, kept as the default for backward compatibility.
- `"ones"`: tied positions become 1 (the mirrored bias).
- `"random"`: tied positions copy a fresh seeded tie-break
  hypervector — the unbiased convention.

`majority(sum_vec, count)` is the shared kernel; the centroid
classifier resolves exact-zero accumulator positions through the same
policy.

## Distance

- The similarity metric is the **Hamming distance**
  `count_nonzero(XOR(a, b))`, an integer in `[0, D]`.
- Normalised distance is `hamming / D` in `[0, 1]`; two independent
  random hypervectors concentrate near `0.5·D` (quasi-orthogonality).
- Distance is symmetric, zero exactly on equality, and bind-invariant:
  `d(a⊗c, b⊗c) = d(a, b)`.

## Clean-up memory tie behaviour

`AssociativeMemory.query` returns the label with the smallest Hamming
distance; on an exact distance tie the **earliest-stored** label wins
(strict `<` comparison over insertion-ordered storage). An empty
memory returns `None`.

## Level encoding

`level_vectors(low, high, levels)` draws level 0 at random and flips
`(D // 2) // (levels − 1)` fresh positions of one fixed seeded
permutation per step, so `d(level_i, level_j) ∝ |i − j|` exactly and
the endpoints approach orthogonality. `encode_level` clips into
`[low, high]` and rounds to the nearest level.

## Relation to external VSA references

The binary {0, 1}/XOR/majority algebra is the classical binary spatter
code (Kanerva). `torchhd`'s BSC model implements the same bind and
majority-bundle semantics over its MAP/BSC tensors; the executed
cross-check in `tests/test_hdc/test_hdc_torchhd_reference.py` compares
bind, bundle (odd counts, where no tie policy is involved), and
permutation against `torchhd` over multiple seeds and dimensions when
that package is installed, and skips cleanly where it is not.
Deliberate differences from `torchhd` defaults:

- default even-count tie handling is the deterministic `"zeros"` bias,
  not a random tie-break — select `tie_policy="random"` for the
  unbiased convention;
- vectors are `uint8` {0, 1} arrays, not torch tensors; the adapter in
  the cross-check test maps between the two rather than weakening
  either semantic.

## Enforcement map

| Rule | Executed test |
|------|---------------|
| Algebraic laws, seed contract, malformed inputs | `tests/test_hdc/test_hdc_algebra_properties.py` |
| Tie policies, level encoding, item memory | `tests/test_hdc/test_hdc_encoder_determinism_and_levels.py` |
| Centroid classifier semantics | `tests/test_hdc/test_hdc_centroid_classifier.py` |
| Retrieval accuracy, collision rate, dimension sweep | `tests/test_hdc/test_hdc_retrieval_regressions.py` |
| `torchhd` reference cross-check (optional dependency) | `tests/test_hdc/test_hdc_torchhd_reference.py` |
| Legacy base behaviour | `tests/test_hdc/test_base.py` |

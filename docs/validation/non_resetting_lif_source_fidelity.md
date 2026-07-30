# Non-resetting LIF dual-identity source and runtime fidelity

This record closes the Model 50 identity correction without destroying the
historical project recurrence.

## Identity decision

The standard/generalized integrate-and-fire and spike-response models in
Jolivet et al. (2004) reset after firing. Kobayashi, Tsubo, and Shinomoto (2009)
instead define the non-resetting MAT family, including one-history MAT(1) with a
50 ms threshold timescale and 2 ms absolute refractory interval. The former
SC-NeuroCore class had non-resetting voltage but neither source reset semantics
nor refractory gating. Therefore:

| Public identity | Meaning | Publication count |
|---|---|---:|
| `NonResettingLIFNeuron` | source MAT(1), non-resetting | one completed source model |
| `SCNonResettingAdaptiveLIFNeuron` | frozen SC affine-rest recurrence | zero additional source models |

## Independent receipts

The source oracle directly transcribes the paper equations without importing a
production kernel. Its 10,272-step drive is 32 zeros, 8,192 samples at 0.7 nA,
then 1,024 repetitions of the alternating `[0.2, 0.9]` pair.

| Observable | Source MAT(1) | Retained SC model |
|---|---|---|
| samples | 10,272 | 256 |
| events | index `[3945]` | count 5 |
| final state | `[27.965935062410335, 32.60279147075955, 0]` | `[-32.61772042832371, -27.97424372241646]` |
| trace SHA-256 | `2ac13e42…2069c6` | `7dd9f76f…3ae25` |

The retained-SC receipt is the frozen pre-split binary64 recurrence, providing a
compatibility proof rather than a publication claim.

## Runtime and benchmark closure

Both identities execute through Python, the modular Rust engine and PyO3 batch
surface, independent Rust safety, Julia, Go, and Mojo. Explicit dispatch fails
closed on unknown or unavailable backends. Events are exact; state traces are
within `2e-12`, with byte-identical Rust/Julia/Go results for both enrolled
200,000-step benchmarks.

| Benchmark | Current | Events | Maximum compiled difference |
|---|---:|---:|---:|
| source MAT(1) | 0.7 nA | 4 | `7.11e-15` |
| retained SC model | 20 | 577 | `2.92e-13` |

Both committed artifacts bind their source files and loaded Rust/Go/Mojo
binaries. They are loaded-host, non-isolated regression evidence and make no
production-speed claim.

## Schema, RTL, and formal closure

Paired TOML/JSON schemas reproduce both hand recurrences. Each hand-written
signed Q32.32 RTL model is cycle-exact to its independent integer oracle,
preserves its enrolled software event vector, synthesizes in Yosys, and matches
the optimized netlist across the checked input sequence. Separate depth-12 CVC5
jobs pass their bounded reset/state/event safety properties.

The optimized-netlist result is sequence-bounded, not universal formal
equivalence. The formal jobs do not establish binary64 equivalence. Timing,
PPA, device, board, and physical-silicon evidence remain open.

## Executed evidence

- `tests/test_model_non_resetting_lif_source_fidelity.py`
- `tests/test_model_sc_non_resetting_adaptive_lif_compatibility.py`
- `tests/test_non_resetting_lif_backend_dispatch.py`
- `tests/test_sc_non_resetting_adaptive_lif_backend_dispatch.py`
- `tests/test_non_resetting_lif_engine_binding.py`
- `tests/test_non_resetting_lif_schema_parity.py`
- `tests/test_sc_non_resetting_adaptive_lif_schema_parity.py`
- `tests/test_reference_non_resetting_lif.py`
- `tests/test_reference_sc_non_resetting_adaptive_lif.py`
- `tests/test_cosim_non_resetting_lif.py`
- `tests/test_cosim_sc_non_resetting_adaptive_lif.py`
- `tests/test_bench_non_resetting_lif_dual_identity.py`
- `hdl/formal/catalogue/sc_non_resetting_lif.sby`
- `hdl/formal/catalogue/sc_non_resetting_adaptive_lif.sby`

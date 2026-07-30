# MAT dual-identity source and runtime fidelity

This evidence record closes the Model 49 identity correction without destroying
the previous project recurrence.

## Identity decision

The cited Kobayashi-Tsubo-Shinomoto MAT* model is non-resetting. Its membrane
obeys `tau_m dV/dt = -V + RI`; its threshold is `omega` plus two exponential
spike-history kernels; and a 2 ms absolute refractory interval suppresses event
emission without freezing or resetting voltage. The former project class used
RK4 and reset voltage. Therefore:

| Public identity | Meaning | Publication count |
|---|---|---:|
| `MATNeuron` | source MAT*, non-resetting | one completed source model |
| `SCResettingMATNeuron` | retained historical SC RK4/reset recurrence | zero additional source models |

`NonResettingLIFNeuron` is not modified by this unit; it remains the immediate
follow-on identity audit.

## Independent source receipt

The direct-equation oracle does not import production kernels. For the paper's
regular-spiking profile it executes 10,272 samples: 32 zeros, 8,192 at 0.7 nA,
then 1,024 `[0.2, 0.9]` pairs. The canonical little-endian row contains four
float64 states plus one uint8 event.

| Observable | Receipt |
|---|---|
| event indices | `[3945]` |
| final state | `[27.965935062410335, 19.654727854241134, 1.9377299916339141, 0]` |
| trace SHA-256 | `3382c1a73215026c1c1f41749cbc2061ff338c044e336e8a25ca59b8ac139de8` |

The SC compatibility oracle separately pins 13 events and SHA-256
`b64411c28f4ab24e87fb52a115fd9379793412350af57a933806f1b6c32af259`.

## Cross-runtime closure

Both identities have complete executable Python, Rust engine, Rust safety,
Julia, Go, and Mojo implementations. The explicit dispatchers fail closed on
unknown/unavailable runtimes. Executed tests compare every voltage, threshold,
refractory (source MAT), and event sample; events are exact and float states are
bounded by `2e-12`.

The source-bound 200,000-step benchmark records 11 source-MAT events under a
constant 0.7 nA drive. Python, Rust, Julia, and Go produce the same trace hash;
Mojo differs only by `7.11e-15` while preserving all events. The separate SC
benchmark records 8,620 events at current 50; Mojo's maximum state difference is
`1.43e-14`. Both artifacts record source and loaded-binary hashes and explicitly
deny production-speed and hardware-measurement claims.

## Schema and silicon closure

Paired TOML/JSON schemas agree with the hand models and their complete enrolled
traces. The RTL boundary is intentionally Q32.32 and bounded:

| Check | Source MAT* | SC resetting MAT |
|---|---:|---:|
| integer-oracle co-simulation | pass | pass |
| Python event-vector parity | pass | pass (13 events) |
| Yosys synthesis/check | pass | pass |
| optimized-netlist enrolled-sequence equivalence | pass | pass |
| CVC5 bounded safety | depth 12 pass | depth 12 pass |

The optimized-netlist result is sequence-bounded, not a universal formal
equivalence theorem. The CVC5 jobs prove only the stated state/event safety
properties under their bounded inputs. No synthesis timing, PPA, foundry,
device, or binary64-equivalence evidence is enrolled.

## Executed evidence

- `tests/test_model_mat_source_fidelity.py`
- `tests/test_model_sc_resetting_mat_compatibility.py`
- `tests/test_mat_backend_dispatch.py`
- `tests/test_sc_resetting_mat_backend_dispatch.py`
- `tests/test_mat_engine_binding.py`
- `tests/test_mat_schema_parity.py`
- `tests/test_sc_resetting_mat_schema_parity.py`
- `tests/test_reference_mat.py`
- `tests/test_reference_sc_resetting_mat.py`
- `tests/test_cosim_mat.py`
- `tests/test_cosim_sc_resetting_mat.py`
- `hdl/formal/catalogue/sc_mat.sby`
- `hdl/formal/catalogue/sc_resetting_mat.sby`

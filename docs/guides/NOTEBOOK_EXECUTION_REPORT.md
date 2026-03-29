# Notebook Execution Report

**Date:** 2026-03-29
**Python:** 3.12.3
**SC-NeuroCore:** 3.14.0
**Platform:** Linux 6.17.0-19-generic x86_64

## Results

| # | Notebook | Status | Output size | Notes |
|---|----------|--------|-------------|-------|
| 08 | equation_to_verilog | PASS | 396 KB | LIF, FHN, Izhikevich → Verilog |
| 09 | topology_and_dynamics | PASS | 495 KB | 6 topologies, raster plots |
| 10 | spike_train_analysis | PASS | 162 KB | ISI, CV, Fano, cross-corr, PCA (with fallback) |
| 11 | biological_circuits | PASS | 245 KB | Tripartite synapse Ca²⁺, Rall dendrite |
| 12 | learning_rules | PASS | 229 KB | STDP, e-prop, R-STDP, STP |
| 13 | quantisation_pipeline | PASS | 166 KB | Float → Q8.8 → SC, error budget |
| 14 | sc_arithmetic_theory | PASS | 399 KB | AND, XNOR, MUX, CORDIV, Sobol, Hoeffding |
| 15 | fault_tolerance | PASS | 275 KB | SC vs FP, stuck-at, TMR |
| 16 | neuron_atlas | PASS | 481 KB | 12 models, voltage traces |
| 17 | reservoir_computing | PASS | 147 KB | LSM, temporal XOR, ridge readout |
| 18 | mixed_precision_sc | PASS | 154 KB | Adaptive per-layer L, Pareto |
| 19 | compression_and_pruning | PASS | 105 KB | Magnitude/SC pruning, quantisation |
| 20 | power_analysis | PASS | 152 KB | Event-driven vs clock-driven |
| 21 | spike_alu | PASS | 16 KB | Gates, register, ALU, sort (text-only) |
| 22 | ir_type_safety | PASS | 13 KB | IR type checker demo (text-only) |
| 23 | topological_observables | PASS | 229 KB | Winding, Ricci, sheaf defect |
| 24 | identity_lazarus | PASS | 43 KB | Checkpoint save/load, TraceEncoder, Director |
| 25 | cortical_column_dynamics | PASS | 99 KB | 5-population canonical microcircuit |
| 26 | spike_codec_benchmark | PARTIAL | 109 KB | Compress cells pass; decompress cells show API errors (allow_errors) |
| 27 | python_to_proven_silicon | PASS | 139 KB | Full ODE→Verilog→testbench pipeline |
| 28 | domain_bridge | PASS | 82 KB | TensorStream, quantum cos²(θ/2) |

**Summary:** 20/21 fully passing, 1 partial (nb26 decompress API differences).

## Test Suite

```
340 passed in 29.48s
```

15 test files, all clean (`ruff check` — 0 errors).

## API Fixes Applied During Execution

| Original (incorrect) | Corrected | Module |
|---------------------|-----------|--------|
| `fano_factor(bin_size=50)` | `fano_factor(window_ms=50.0)` | variability |
| `cross_correlation(max_lag=50)` | `cross_correlation(max_lag_ms=50.0)` | correlation |
| `pairwise_correlation(a, b, bin_size)` | `pairwise_correlation([a, b, ...])` | correlation |
| `sttc(dt_window=5)` | `sttc(delta_ms=5.0)` | correlation |
| `van_rossum_distance(st, st, tau=0.01)` | `van_rossum_distance(train, train, tau_ms=10.0)` | distance |
| `victor_purpura(st, st, q=100)` | `victor_purpura(times, times, cost_per_s=1000)` | distance |
| `rng=np.random.default_rng(42)` | `rng=RNG(42)` | bitstreams |
| `ShortTermPlasticity(U=0.15, base_weight=1.0)` | `ShortTermPlasticity(u_se=0.15)` | learning |
| `stp.step(spike, dt=1.0)` | `stp.update(pre_spikes_array)` | learning |
| `CorticalColumn.run(matrix, steps)` | `CorticalColumn.run(constant_vector, steps)` | network |
| `codec.decompress(data)` | `codec.decompress(data, T, N)` | codecs (ISI/Predictive/Delta) |
| `encoded:10s` | `encoded:#10x` | Q8.8 format string |

## Total Output

- 21 notebooks with cell outputs: **4.38 MB** total
- 15 test files: **2,742 lines**, 340 tests
- 1 documentation guide: `docs/guides/notebook_guide.md`
- 7 tutorials updated with notebook cross-references

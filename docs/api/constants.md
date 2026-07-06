# Named Physical and Model Constants

**Module:** `sc_neurocore.constants`
**Source:** `src/sc_neurocore/constants.py` — 103 LOC, 44 module-level
constants
**Status (v3.16.0):** module exists with 44 named constants. All 44 are
imported by the 16 maintained Python source modules listed in §3 and guarded
by `tests/test_constants.py`. Task #37 is closed for Python adoption honesty,
dedicated tests, and the Izhikevich threshold comment. Rust parity automation
remains a separate open follow-up (§7.3, task #38).

This page lists every constant, its provenance citation, and which
ones are actually used.

---

## 1. Purpose (per the module docstring)

> This module is the citation ledger for 44 named constants that trace
> to textbook references, hardware design choices, or documented
> conventions. Current adoption is explicit and tested: 16 maintained
> source modules import all 44 named constants from this ledger. See
> `docs/api/constants.md` for the Python import boundary and the
> separate Rust parity note.

The purpose is a single auditable Python source for model defaults and
hardware-facing constants. The current guarantee is deliberately scoped:
Python source adoption is tested, while Rust-side values are still maintained
manually and tracked separately as task #38.

---

## 2. The 44 constants by group

### 2.1 LIF neuron defaults — Gerstner & Kistler 2002

> Gerstner W., Kistler W. M. *Spiking Neuron Models: Single Neurons,
> Populations, Plasticity*. Cambridge University Press (2002).
> Table 1.1, normalised units.

| Name | Value | Unit |
|------|------:|------|
| `LIF_V_REST` | `0.0` | mV (normalised) |
| `LIF_V_RESET` | `0.0` | mV (normalised) |
| `LIF_V_THRESHOLD` | `1.0` | mV (normalised) |
| `LIF_TAU_MEM` | `20.0` | ms — Ch. 1 |
| `LIF_RESISTANCE` | `1.0` | MΩ (normalised) |
| `LIF_DT` | `1.0` | ms |
| `LIF_NOISE_STD` | `0.0` | mV |
| `LIF_REFRACTORY_PERIOD` | `0` | ms |
| `LIF_LAYER_NOISE_STD` | `0.02` | mV |

### 2.2 Izhikevich neuron — Izhikevich 2003 ("Regular Spiking")

> Izhikevich E. M. "Simple Model of Spiking Neurons." *IEEE Trans
> Neural Networks* 14(6):1569-1572 (2003). Table 1.

| Name | Value | Unit |
|------|------:|------|
| `IZH_A` | `0.02` | 1/ms |
| `IZH_B` | `0.2` | 1/ms |
| `IZH_C` | `-65.0` | mV |
| `IZH_D` | `8.0` | mV |
| `IZH_SPIKE_THRESHOLD` | `30.0` | mV spike-detect peak, not an adaptive membrane threshold |

### 2.3 Homeostatic threshold adaptation — Turrigiano 2012

> Turrigiano G. "Homeostatic Synaptic Plasticity." *Cold Spring
> Harbor Perspectives in Biology* 4:a005736 (2012).

| Name | Value | Unit |
|------|------:|------|
| `HOMEOSTATIC_TARGET_RATE` | `0.1` | spikes/step |
| `HOMEOSTATIC_ADAPTATION_RATE` | `0.01` | dimensionless |
| `HOMEOSTATIC_TRACE_DECAY` | `0.95` | dimensionless |
| `HOMEOSTATIC_THRESHOLD_FLOOR` | `0.1` | mV |
| `HOMEOSTATIC_THRESHOLD_CEILING_MULT` | `10.0` | × initial |

### 2.4 Dendritic XOR — Koch 1999

> Koch C. *Biophysics of Computation*. Oxford UP (1999), Ch. 12.

| Name | Value | Unit |
|------|------:|------|
| `DENDRITIC_THRESHOLD` | `0.5` | dimensionless |

### 2.5 Fixed-point hardware — matches `sc_lif_neuron.v` RTL

| Name | Value | Note |
|------|------:|------|
| `FP_DATA_WIDTH` | `16` | Q8.8 total width |
| `FP_FRACTION` | `8` | fractional bits in Q8.8 |
| `FP_V_THRESHOLD` | `256` | Q8.8 representation of 1.0 |
| `FP_REFRACTORY_PERIOD` | `2` | clock cycles |
| `FP_LFSR_WIDTH` | `16` | bits |
| `FP_LFSR_SEED` | `0xACE1` | maximal-length LFSR seed |

### 2.6 STDP and reward-modulated STDP — Bi & Poo 1998 / Izhikevich 2007

> Bi G., Poo M. "Synaptic Modifications in Cultured Hippocampal
> Neurons." *J Neurosci* 18(24):10464-10472 (1998), Fig. 1.
>
> Izhikevich E. M. "Solving the Distal Reward Problem through
> Linkage of STDP and Dopamine Signaling." *Cerebral Cortex*
> 17(10):2443-2452 (2007).

| Name | Value | Note |
|------|------:|------|
| `STDP_LEARNING_RATE` | `0.01` | dimensionless |
| `STDP_WINDOW_SIZE` | `5` | timesteps |
| `STDP_LTD_RATIO` | `0.5` | LTD/LTP asymmetry |
| `SYNAPSE_DEFAULT_LENGTH` | `256` | bits |
| `SYNAPSE_DEFAULT_WEIGHT` | `0.5` | dimensionless |
| `RSTDP_TRACE_DECAY` | `0.9` | dimensionless |
| `RSTDP_ANTI_HEBBIAN_SCALE` | `0.5` | dimensionless |

### 2.7 Layer defaults

| Name | Value | Unit |
|------|------:|------|
| `LAYER_DEFAULT_LENGTH` | `1024` | bits |
| `LAYER_CONV_LENGTH` | `256` | bits |
| `DENSE_LAYER_LENGTH` | `2048` | bits |
| `DENSE_Y_MIN` | `0.0` | current source lower bound |
| `DENSE_Y_MAX` | `0.1` | current source upper bound |

### 2.8 Reservoir / echo-state — Jaeger 2001

> Jaeger H. "The Echo State Approach to Analysing and Training
> Recurrent Neural Networks." *GMD Report* 148 (2001).

| Name | Value | Note |
|------|------:|------|
| `RESERVOIR_FEEDBACK_STRENGTH` | `0.5` | dimensionless |
| `RESERVOIR_INPUT_STRENGTH` | `0.5` | dimensionless |
| `RESERVOIR_SPECTRAL_RADIUS` | `0.9` | echo state property bound |

### 2.9 Memristive crossbar — Prezioso et al. 2015

> Prezioso M. *et al.* "Training and operation of an integrated
> neuromorphic network based on metal-oxide memristors." *Nature*
> 521:61-64 (2015).

| Name | Value | Note |
|------|------:|------|
| `MEMRISTIVE_STUCK_RATE` | `0.01` | fraction stuck at 0 or 1 |
| `MEMRISTIVE_VARIABILITY` | `0.05` | σ of conductance drift |

### 2.10 RNG seed offset

| Name | Value | Note |
|------|------:|------|
| `NEURON_SEED_OFFSET` | `10_000` | separates neuron from synapse RNG streams |

---

## 3. Honesty notice — current Python adoption boundary

The earlier v3.14 audit reported one Python consumer and 36 unused constants.
That is stale in the current tree. As of v3.16.0, `tests/test_constants.py`
parses maintained source under `src/sc_neurocore`, excludes nested build/cache
trees such as `.pixi`, and asserts this import boundary:

| Source file | Imported constants |
|-------------|--------------------|
| `src/sc_neurocore/layers/fusion.py` | `LAYER_DEFAULT_LENGTH` |
| `src/sc_neurocore/layers/jax_dense_layer.py` | `LAYER_DEFAULT_LENGTH`, `LIF_DT`, `LIF_LAYER_NOISE_STD`, `LIF_RESISTANCE`, `LIF_TAU_MEM`, `LIF_V_RESET`, `LIF_V_REST`, `LIF_V_THRESHOLD` |
| `src/sc_neurocore/layers/memristive.py` | `MEMRISTIVE_STUCK_RATE`, `MEMRISTIVE_VARIABILITY` |
| `src/sc_neurocore/layers/recurrent.py` | `LAYER_DEFAULT_LENGTH`, `RESERVOIR_FEEDBACK_STRENGTH`, `RESERVOIR_INPUT_STRENGTH`, `RESERVOIR_SPECTRAL_RADIUS` |
| `src/sc_neurocore/layers/sc_conv_layer.py` | `LAYER_CONV_LENGTH` |
| `src/sc_neurocore/layers/sc_dense_layer.py` | `DENSE_LAYER_LENGTH`, `DENSE_Y_MAX`, `DENSE_Y_MIN`, `LIF_DT`, `LIF_LAYER_NOISE_STD`, `LIF_RESISTANCE`, `LIF_TAU_MEM`, `LIF_V_RESET`, `LIF_V_REST`, `LIF_V_THRESHOLD`, `NEURON_SEED_OFFSET` |
| `src/sc_neurocore/layers/sc_learning_layer.py` | `LAYER_DEFAULT_LENGTH`, `STDP_LEARNING_RATE`, `STDP_LTD_RATIO` |
| `src/sc_neurocore/layers/vectorized_layer.py` | `LAYER_DEFAULT_LENGTH` |
| `src/sc_neurocore/neurons/dendritic.py` | `DENDRITIC_THRESHOLD` |
| `src/sc_neurocore/neurons/fixed_point_lif.py` | `FP_DATA_WIDTH`, `FP_FRACTION`, `FP_LFSR_SEED`, `FP_LFSR_WIDTH`, `FP_REFRACTORY_PERIOD`, `FP_V_THRESHOLD` |
| `src/sc_neurocore/neurons/homeostatic_lif.py` | `HOMEOSTATIC_ADAPTATION_RATE`, `HOMEOSTATIC_TARGET_RATE`, `HOMEOSTATIC_THRESHOLD_CEILING_MULT`, `HOMEOSTATIC_THRESHOLD_FLOOR`, `HOMEOSTATIC_TRACE_DECAY` |
| `src/sc_neurocore/neurons/sc_izhikevich.py` | `IZH_A`, `IZH_B`, `IZH_C`, `IZH_D`, `IZH_SPIKE_THRESHOLD`, `LIF_DT` |
| `src/sc_neurocore/neurons/stochastic_lif.py` | `LIF_DT`, `LIF_NOISE_STD`, `LIF_REFRACTORY_PERIOD`, `LIF_RESISTANCE`, `LIF_TAU_MEM`, `LIF_V_RESET`, `LIF_V_REST`, `LIF_V_THRESHOLD` |
| `src/sc_neurocore/synapses/r_stdp.py` | `RSTDP_ANTI_HEBBIAN_SCALE`, `RSTDP_TRACE_DECAY` |
| `src/sc_neurocore/synapses/sc_synapse.py` | `SYNAPSE_DEFAULT_LENGTH`, `SYNAPSE_DEFAULT_WEIGHT` |
| `src/sc_neurocore/synapses/stochastic_stdp.py` | `STDP_LEARNING_RATE`, `STDP_LTD_RATIO`, `STDP_WINDOW_SIZE` |

Together these consumers import all 44 names. The guarantee is a Python source
guarantee only; Rust engine parity remains manual until task #38 adds an
automated sync or parity gate.

---

## 4. Recommended path forward

Task #37 is closed for the Python constants ledger:

1. The module docstring now states the current 16-module / 44-constant
   adoption boundary instead of a blanket codebase-wide claim.
2. `tests/test_constants.py` guards exact values, scalar types, physical
   ranges, Q8.8 self-consistency, the source import map, and the Izhikevich
   spike-threshold comment.
3. The scoped public docstring policy now includes `src/sc_neurocore/constants.py`.

The next related task is #38: add an automated Python/Rust parity gate for
constants mirrored in `engine/src/` before any future cross-language value
change is accepted.

---

## 5. Pipeline wiring

| Surface | How it's wired | Verifier |
|---------|---------------|----------|
| Python constants ledger | flat module export from `sc_neurocore.constants` | `tests/test_constants.py::test_public_constant_ledger_matches_documented_values` |
| Python source adoption map | direct imports from 16 maintained modules | `tests/test_constants.py::test_source_import_map_matches_current_adoption_boundary` |
| Physical range and Q8.8 invariants | value-level assertions | `tests/test_constants.py::test_dimensionless_constants_remain_in_physical_ranges`; `tests/test_constants.py::test_fixed_point_q88_constants_are_self_consistent` |
| Rust engine parity | manual review only | task #38 |

---

## 6. Audit (7-point checklist)

| # | Dimension | Status | Detail |
|---|-----------|--------|--------|
| 1 | Pipeline wiring | ✅ PASS | 16 maintained Python source modules import all 44 constants; the import map is tested. |
| 2 | Multi-angle tests | ✅ PASS | `tests/test_constants.py` asserts exact values, scalar types, ranges, Q8.8 consistency, source import adoption, the module docstring boundary, and the Izhikevich threshold comment. |
| 3 | Rust path | ⚠️ WARN | Plain Python data; the Rust engine has its own constants in `engine/src/`. The two need to be kept in sync manually — if `LIF_V_THRESHOLD` here changes, `engine/src/neurons/lif.rs` must update too. No automated check exists. |
| 4 | Benchmarks | N/A | Constants do not run code |
| 5 | Performance docs | N/A | Same |
| 6 | Documentation page | ✅ PASS | This page |
| 7 | Rules followed | ✅ PASS | SPDX header ✅. Module docstring is scoped to the tested Python adoption boundary. Citations are present. British English clean. No `# noqa`, no `# type: ignore`. |

Net: **1 WARN, 0 FAIL.** The remaining WARN is the open Rust parity
automation gap tracked as task #38.

---

## 7. Known issues

### 7.1 Adoption gap (task #37)

Closed for maintained Python source. The current import map covers all 44
constants across the 16 source consumers listed in §3.

### 7.2 Dedicated constants tests

Closed. `tests/test_constants.py` now provides direct tests for:

- exact 44-value ledger contents and scalar types;
- normalised ranges and positive physical parameters;
- fixed-point Q8.8 self-consistency;
- maintained-source import boundary;
- module docstring honesty;
- Izhikevich spike-detect threshold wording.

### 7.3 Rust-side constants are not synchronised

`engine/src/neurons/lif.rs` (and similar files) contain their own
copies of the LIF defaults (`tau_mem = 20.0`, `v_threshold = 1.0`,
etc.). If this Python file changes, the Rust file must change too —
there is no build-time check that the two agree. A
`build.rs`-style sync or a runtime parity test (Python value ==
Rust value via PyO3) would close this.

Tracked as task #38.

### 7.4 `IZH_SPIKE_THRESHOLD = 30.0` is the spike-detect peak,
not an adaptive membrane threshold

Closed. The source comment now identifies `30.0` mV as the spike-detect peak
and explicitly says it is not an adaptive membrane threshold.

---

## 8. Tests

`tests/test_constants.py` is the dedicated regression suite for this module.
It is intentionally source-aware: the adoption-boundary test parses maintained
Python source instead of relying on a free-text grep, so nested build/cache
trees do not pollute the evidence.

---

## 9. References

All citations are inline in the source file and reproduced in §2 by
group. The primary references:

- Gerstner W., Kistler W. M. *Spiking Neuron Models* (2002)
- Izhikevich E. M. *IEEE TNN* 14(6):1569-1572 (2003)
- Turrigiano G. *Cold Spring Harb Perspect Biol* 4:a005736 (2012)
- Koch C. *Biophysics of Computation* (1999)
- Bi G., Poo M. *J Neurosci* 18(24):10464-10472 (1998)
- Izhikevich E. M. *Cerebral Cortex* 17(10):2443-2452 (2007)
- Jaeger H. *GMD Report* 148 (2001)
- Prezioso M. *et al.* *Nature* 521:61-64 (2015)

Internal:

- Compiler Q8.8 hard-coding: [`api/cli.md`](cli.md), §9.1
- Exception hierarchy (parallel "declared but unused" pattern):
  [`api/exceptions.md`](exceptions.md), §4

---

## 10. Auto-rendered API

::: sc_neurocore.constants
    options:
      show_root_heading: true
      show_source: true

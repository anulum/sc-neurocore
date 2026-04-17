# Named Physical and Model Constants

**Module:** `sc_neurocore.constants`
**Source:** `src/sc_neurocore/constants.py` — 100 LOC, 44 module-level
constants
**Status (v3.14.0):** module exists with 44 named constants; **only
1 of the ~250 source files in `src/` actually imports from it**, and
only 8 of the 44 constants are referenced anywhere outside the module.
The docstring claims "Modules import from here instead of using bare
numeric literals" but this convention is not adopted in the wider
codebase (§3 honesty notice).

This page lists every constant, its provenance citation, and which
ones are actually used.

---

## 1. Purpose (per the module docstring)

> Every default parameter in the library traces back to either a
> textbook reference, a hardware design choice, or an explicitly
> documented convention. Modules import from here instead of using
> bare numeric literals.

The intent is sound: a single audit-able source for every default
in the library, with each value traceable to a publication or a
design decision. The implementation matches the intent inside this
file (every value carries a comment naming its source). What is
not implemented is the second sentence — adoption across the rest
of `src/`.

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
| `IZH_SPIKE_THRESHOLD` | `30.0` | mV |

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

## 3. Honesty notice — adoption gap

The module docstring states:

> Modules import from here instead of using bare numeric literals.

In v3.14.0 this is **not the case**. A grep across `src/` finds only
**one file** that imports from `sc_neurocore.constants`:

```
src/sc_neurocore/layers/jax_dense_layer.py
```

That file imports 8 of the 44 constants:
`LAYER_DEFAULT_LENGTH`, `LIF_DT`, `LIF_V_REST`, `LIF_V_RESET`,
`LIF_V_THRESHOLD`, `LIF_TAU_MEM`, `LIF_RESISTANCE`,
`LIF_LAYER_NOISE_STD`.

The other **36 constants are declared but unused outside this
module**. Examples:

- `IZH_A/B/C/D` — duplicated as inline literals in
  `src/sc_neurocore/neurons/models/izhikevich.py`
- `STDP_LEARNING_RATE` — duplicated in synapse / plasticity files
- `FP_DATA_WIDTH`, `FP_FRACTION` — the equation compiler hard-codes
  16 and 8 instead (`compiler/equation_compiler.py:39-40`)
- `LIF_TAU_MEM = 20.0` — inline in `LapicqueNeuron(tau=20.0)` and
  many other neuron defaults

This is **not behavioural breakage** — the constants in `constants.py`
agree with the inline literals — but it does mean:

1. Updating `LIF_TAU_MEM` in `constants.py` does **not** change the
   behaviour of any neuron model unless the model is rebuilt to
   import it.
2. Auditors cannot trust that the citation comments in this file
   actually describe the running code; they describe the module's
   **intended** values.

Tracked as task #37.

---

## 4. Recommended path forward

Three options, in order of effort:

1. **Document the reserved-vocabulary status.** Add a paragraph to
   the module docstring stating "v3.14.0 status: only LIF defaults
   are imported by `jax_dense_layer.py`; the other 36 constants
   document intent but are not currently the source of truth for
   any module. See `docs/api/constants.md` §3."

2. **Migrate inline literals to constants imports.** For each
   module that uses a value declared here, replace the literal with
   the named import. Order of effort:
   - LIF defaults: handful of files
   - Izhikevich: 1 file
   - Homeostatic: 1-2 files
   - FP_ hardware: compiler + RTL
   - STDP/plasticity: ~5 files
   - Reservoir: 1 file
   - Memristive: 1 file
   The migration is mechanical but touches ~12-15 files. Each
   change must be a separate commit per task-completion protocol.

3. **Delete the unused 36.** Risky — they document intent. Only do
   this if option 2 is rejected and option 1 is judged misleading.

Tracked as task #37.

---

## 5. Pipeline wiring

| Surface | How it's wired | Verifier |
|---------|---------------|----------|
| `from sc_neurocore.constants import LIF_V_THRESHOLD, ...` | flat module export | `src/sc_neurocore/layers/jax_dense_layer.py` (only consumer) |
| Other 36 constants | declared but no importers | n/a |

---

## 6. Audit (7-point checklist)

| # | Dimension | Status | Detail |
|---|-----------|--------|--------|
| 1 | Pipeline wiring | ⚠️ WARN | Only 1 consumer of 8 constants; 36 constants declared with no consumer in src/ |
| 2 | Multi-angle tests | ❌ FAIL | **No `tests/test_constants.py` file exists.** The constants change project behaviour only via `jax_dense_layer.py` (covered transitively); no test asserts the citation values themselves. |
| 3 | Rust path | N/A | Plain Python data; the Rust engine has its own constants in `engine/src/`. The two need to be kept in sync manually — if `LIF_V_THRESHOLD` here changes, `engine/src/neurons/lif.rs` must update too. No automated check exists. |
| 4 | Benchmarks | N/A | Constants do not run code |
| 5 | Performance docs | N/A | Same |
| 6 | Documentation page | ✅ PASS | This page |
| 7 | Rules followed | ⚠️ WARN | SPDX header ✅. **Module docstring overstates adoption** (§3). Citations are present and correct as far as `git blame` shows. British English clean. No `# noqa`, no `# type: ignore`. |

Net: **2 WARN, 1 FAIL.** The FAIL is a missing test file; the WARNs
are the docstring overstatement and the orphan-constant adoption
gap.

---

## 7. Known issues

### 7.1 Adoption gap (task #37)

See §3-§4. This is the headline issue.

### 7.2 No `tests/test_constants.py`

Per `feedback_test_sophistication`: "STRONG is the minimum default".
A constants module with 44 declared values warrants at least:
- Type assertion (every constant has the documented dtype)
- Range assertion (e.g. `0 < HOMEOSTATIC_TARGET_RATE < 1`)
- Unit-citation cross-check (does each constant's value match its
  comment's claimed paper?)

The cross-check would catch silent drift if someone edits a
constant without updating its citation. Tracked as part of task #37.

### 7.3 Rust-side constants are not synchronised

`engine/src/neurons/lif.rs` (and similar files) contain their own
copies of the LIF defaults (`tau_mem = 20.0`, `v_threshold = 1.0`,
etc.). If this Python file changes, the Rust file must change too —
there is no build-time check that the two agree. A
`build.rs`-style sync or a runtime parity test (Python value ==
Rust value via PyO3) would close this.

Tracked as task #38.

### 7.4 `IZH_SPIKE_THRESHOLD = 30.0` is the spike-detect threshold,
not the membrane threshold

A reader scanning the file might assume `IZH_SPIKE_THRESHOLD`
governs when an Izhikevich neuron fires. It actually marks the
**post-spike voltage** at which the spike is detected and reset
applied — a distinction Izhikevich 2003 makes explicit but the
constant name does not. Add a one-line comment.

---

## 8. Tests

There is no `tests/test_constants.py`. The constants are exercised
indirectly via `tests/test_layers.py` (which tests `JaxSCDenseLayer`,
the only consumer).

Recommended minimal test (not yet written):

```python
# tests/test_constants.py
from sc_neurocore import constants


def test_lif_threshold_is_normalised():
    assert constants.LIF_V_THRESHOLD == 1.0


def test_izh_spike_threshold_matches_paper():
    # Izhikevich 2003 Table 1: v_peak = +30 mV
    assert constants.IZH_SPIKE_THRESHOLD == 30.0


def test_fp_q88_is_self_consistent():
    assert constants.FP_DATA_WIDTH == 16
    assert constants.FP_FRACTION == 8
    assert constants.FP_V_THRESHOLD == 1 << constants.FP_FRACTION


def test_homeostatic_target_rate_in_unit_interval():
    assert 0 < constants.HOMEOSTATIC_TARGET_RATE < 1
```

Tracked as part of task #37.

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

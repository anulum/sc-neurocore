# Cortical Column Microcircuit

**Module:** `sc_neurocore.network.cortical_column`
**Source:** `src/sc_neurocore/network/cortical_column.py` — 176 LOC,
single `CorticalColumn` dataclass
**Status (v3.14.0):** simplified 5-population canonical-microcircuit
sketch. The class **cites** Douglas & Martin 2004 and
Potjans & Diesmann 2014 in its module docstring, but the implementation
**does not reproduce** either paper. This page documents the cited
specifications, the actual implementation, the gap between them, and the
empirical dynamics of the current code. A fidelity-restoration follow-up
is tracked as task #10.

> **Honesty notice.** Read sections [§4 Gap Analysis](#4-gap-analysis-vs-cited-papers)
> and [§5 Empirical Dynamics](#5-empirical-dynamics-of-the-current-implementation)
> before relying on this code for anything that claims to match Potjans
> & Diesmann. The current implementation produces firing rates 30×–1000×
> the published values and has three fewer populations than the paper.

---

## 1. What the cited papers specify

### 1.1 Douglas & Martin 2004 — *Neuronal circuits of the neocortex*

*Annu Rev Neurosci* 27:419-451 (2004). A review article describing the
**canonical cortical microcircuit** as a 6-layer architecture with three
broad signal pathways:

- L4 receives thalamic input
- L4 → L2/3 → L5 / L6 (the core feed-forward chain)
- L5 → L6 → thalamus (the corticothalamic feedback)
- Recurrent local connectivity within and across each layer
- Inhibitory interneurons in every layer (PV+ basket cells dominate L2/3
  and L4; SST+ Martinotti cells dominate L5/L6)

Douglas & Martin do not specify a connection matrix or numerical
parameters; they describe the *shape* of the circuit. Reproducing them
means having (a) at least 6 layers with both excitatory and inhibitory
populations, and (b) the three signal pathways above.

### 1.2 Potjans & Diesmann 2014 — *The cell-type specific cortical microcircuit*

*Cerebral Cortex* 24(3):785-806 (2014). The **quantitative** canonical
column. This paper specifies:

- **8 populations** — L2/3 excitatory + inhibitory, L4 exc + inh, L5 exc
  + inh, L6 exc + inh.
- **Population sizes** for a 1 mm² column (Table 1):
  L23E 20683, L23I 5834, L4E 21915, L4I 5479,
  L5E 4850, L5I 1065, L6E 14395, L6I 2948 — total ≈ 77,169 neurons.
- **Connection probability matrix** (Table 5, the
  Binzegger-derived 8×8) with all 64 entries specified. Self-projection
  probabilities range 0.000 to 0.156.
- **Synaptic weights**: PSP amplitudes uniformly 0.15 mV (L4E→L23E gets
  the special 0.30 mV "feedforward" boost).
- **Conduction delays** drawn from Gaussian distributions
  (mean 1.5 ms exc, 0.75 ms inh; SD 0.75 ms / 0.375 ms; clipped at 0.1 ms).
- **External Poisson input** at population-specific rates per neuron
  (Table 5 right column) — this is the **dominant driver** of the
  background activity; without it the network is silent.
- **PSP kernel**: exponential current-based synapses with
  `tau_syn = 0.5 ms`, integrated through a leaky IF membrane with
  `tau_m = 10 ms`, `V_th = -50 mV`, `V_reset = -65 mV`,
  `t_ref = 2 ms`.
- **Target firing rates** (Table 4, "asynchronous irregular" baseline):
  L23E 0.86 Hz, L23I 2.94 Hz, L4E 4.45 Hz, L4I 5.83 Hz, L5E 7.59 Hz,
  L5I 8.27 Hz, L6E 1.10 Hz, L6I 7.66 Hz.

Reproducing Potjans & Diesmann means producing those specific rates
under the specific connectivity, weights, delays and external input.

---

## 2. What this implementation has

`CorticalColumn` is a dataclass with these defaults:

```python
@dataclass
class CorticalColumn:
    n_per_layer: int = 20
    tau: float = 10.0
    dt: float = 1.0
    w_exc: float = 0.1
    w_inh: float = -0.15
    threshold: float = 1.0
    seed: int | None = None
```

### 2.1 Populations: 5 of the 8 in Potjans

| Implementation | Variable | Potjans population covered | Potjans population MISSING |
|----------------|----------|----------------------------|------------------------------|
| L2/3 excitatory | `v_l23_exc` | L23E | — |
| L2/3 inhibitory | `v_l23_inh` | L23I | — |
| L4 (no inh split) | `v_l4` | L4E | **L4I (~5479 neurons)** |
| L5 (no inh split) | `v_l5` | L5E | **L5I (~1065 neurons)** |
| L6 (no inh split) | `v_l6` | L6E | **L6I (~2948 neurons)** |

The three missing inhibitory populations matter: in Potjans the deep-layer
inhibition is what stabilises L5 and L6. Without it, L5/L6 in this
implementation either silence (default weights) or run away.

### 2.2 Connectivity: 7 of the 64 in the Binzegger 8×8

The constructor builds 7 weight matrices (`__post_init__`,
`cortical_column.py:80-98`):

| Edge | Probability | Strength | Potjans equivalent |
|------|-------------|----------|--------------------|
| `thal → L4` | 0.5 | `+w_exc` | external Poisson input (no fixed connectivity) |
| `L4 → L23E` | 0.4 | `+w_exc` | L4E → L23E (0.0838 in paper) |
| `L23E → L23I` | 0.3 | `+w_exc` | L23E → L23I (0.1346) |
| `L23I → L23E` | 0.3 | `+w_inh` | L23I → L23E (0.1346) |
| `L23E → L5` | 0.3 | `+w_exc` | L23E → L5E (0.0203) |
| `L5 → L6` | 0.3 | `+w_exc` | L5E → L6E (0.0090) |
| `L6 → L4` | 0.2 | `+w_exc * 0.5` | L6E → L4E (0.0156) |

The 64-entry Binzegger matrix specifies many additional edges that this
code does not have, including:

- **L23E → L23E** recurrent excitation (paper p=0.1009)
- **All deep-layer inhibitory connections** (since L4I/L5I/L6I are absent)
- **Cross-layer inhibition** like L4I → L23E (paper p=0.0691) or
  L5I → L23E (p=0.0364)
- **L4E → L4E recurrent** (paper p=0.0497)
- **L5E → L5E recurrent** (paper p=0.0831)

Probabilities are also order-of-magnitude different — the implementation
uses 0.2–0.5 (dense), the paper uses 0.005–0.16 (sparse).

### 2.3 Membrane model: simplified LIF without PSP kernel

Each layer is a vector of leaky integrator voltages updated as:

```python
self.v_l4 = self._decay * self.v_l4 + i_l4 * self.dt / self.tau
spk_l4 = (self.v_l4 > self.threshold).astype(float)
self.v_l4 -= spk_l4 * self.threshold
```

Differences from Potjans:

- **No PSP kernel** — incoming spikes are treated as instantaneous
  current pulses scaled by `w_exc` / `w_inh`. The paper integrates each
  spike through an exponential `tau_syn = 0.5 ms` synaptic kernel.
- **No refractory period** — Potjans uses `t_ref = 2 ms`, this code none.
- **No conduction delays** — Potjans samples from
  `N(1.5 ms, 0.75 ms)` for excitatory connections; this code's matvec
  is instantaneous.
- **No biological units** — `threshold = 1.0` is dimensionless; Potjans
  uses `V_th = -50 mV, V_reset = -65 mV`.
- **`dt = 1.0` (default in this class) is a unit-free step**; Potjans
  uses `dt = 0.1 ms`.

### 2.4 External drive: only thalamic input to L4

`step(thalamic_input)` accepts a single vector of length `n_per_layer`
and routes it through `w_thal_to_l4`. There is **no Poisson background
input** to any layer. Potjans explicitly notes that the asynchronous
irregular regime depends on the per-population background rates; without
them the network is essentially silent (or hyper-driven, depending on
weights).

---

## 3. Public surface

```python
@dataclass
class CorticalColumn:
    n_per_layer: int = 20
    tau: float = 10.0          # leak time constant (units: same as dt)
    dt: float = 1.0            # timestep
    w_exc: float = 0.1         # excitatory connection strength
    w_inh: float = -0.15       # inhibitory connection strength
    threshold: float = 1.0     # spike threshold
    seed: int | None = None    # RNG seed for connectivity matrices

    def step(self, thalamic_input: np.ndarray) -> dict[str, np.ndarray]:
        """One timestep. Returns {'l23_exc','l23_inh','l4','l5','l6': spikes}."""

    def run(self, thalamic_input: np.ndarray, steps: int) -> dict[str, np.ndarray]:
        """Repeated step() with constant input. Returns same keys, shape (steps, n)."""

    def reset(self) -> None:
        """Zero all membrane voltages."""
```

`step` returns a `dict` with five keys (`l23_exc`, `l23_inh`, `l4`,
`l5`, `l6`). Each value is a binary spike vector of shape
`(n_per_layer,)`.

`run` calls `step` `steps` times with the same input and stacks results
into `(steps, n_per_layer)` arrays.

---

## 4. Gap analysis vs cited papers

| Aspect | Douglas & Martin 2004 | Potjans & Diesmann 2014 | This code | Gap |
|--------|------------------------|--------------------------|-----------|-----|
| Populations | 6 layers × {E, I} = 12 conceptually | 8 populations specified | 5 populations | Missing L4I, L5I, L6I |
| Population sizes | not specified numerically | 1065 – 21915 (Table 1) | configurable, default 20 | 3 orders of magnitude smaller |
| Connection topology | qualitative diagram | Binzegger 8×8 matrix (64 entries) | 7 hand-picked edges | 57/64 entries missing |
| Connection probabilities | not specified | 0.005 – 0.16 (sparse) | 0.2 – 0.5 (dense) | order of magnitude wrong |
| Synaptic weights | not specified | 0.15 mV PSP (uniform), 0.30 mV (L4E→L23E only) | scalar `w_exc=0.1`, `w_inh=-0.15` | dimensionless, no PSP shape |
| Synaptic kernel | not specified | exponential `tau_syn = 0.5 ms` | none (instantaneous) | absent |
| Conduction delays | exist | `N(1.5 ms, 0.75 ms)` exc, `N(0.75 ms, 0.375 ms)` inh | none | absent |
| Refractory period | exists biologically | `t_ref = 2 ms` | none | absent |
| Membrane model | not specified | LIF, `tau_m=10 ms`, `V_th=-50 mV`, `V_reset=-65 mV` | LIF dimensionless, `tau=10`, `threshold=1.0` | unitless |
| External drive | thalamus + cortico-cortical | per-population Poisson, rates from Table 5 | only thalamic input to L4 | no background |
| Verified output | qualitative pathway diagram | Table 4 firing rates (0.86–8.27 Hz) | sweep below | rates are 30×–1000× the paper |

**Net:** the implementation is best described as a *5-population
feedforward sketch loosely inspired by* the canonical microcircuit, not
as a reproduction of either paper. Removing the Potjans citation from
the module docstring or restoring fidelity is tracked as task #10.

---

## 5. Empirical dynamics of the current implementation

Direct measurements on this workstation, not extrapolations.

### 5.1 Default parameters, single drive amplitude

```python
col = CorticalColumn(n_per_layer=20, seed=42, threshold=1.0,
                     w_exc=0.1, w_inh=-0.15)
res = col.run(np.ones(20) * 5.0, steps=1000)
# 39.0 ms wall
```

Per-layer firing rate (assuming `dt = 1 ms`):

| Layer | Spikes / 1000 steps | Rate (Hz) |
|-------|--------------------:|----------:|
| `l4` | 3 944 | 197.2 |
| `l23_exc` | 0 | 0.0 |
| `l23_inh` | 0 | 0.0 |
| `l5` | 0 | 0.0 |
| `l6` | 0 | 0.0 |

L4 saturates at ~200 Hz; nothing propagates downstream. The L4 → L23E
weight (`w_exc = 0.1`) is too weak to drive L23E above threshold.

### 5.2 Weight / drive sweep

```python
for w in [0.1, 0.3, 0.5, 1.0]:
    for drive in [3.0, 5.0]:
        col = CorticalColumn(n_per_layer=20, seed=42,
                             threshold=1.0, w_exc=w, w_inh=-0.15)
        rates = col.run(np.ones(20) * drive, steps=500)  # → Hz @ dt=1 ms
```

Resulting per-layer rates (Hz):

| `w_exc` | `thal` | L4 | L23E | L23I | L5 | L6 |
|--------:|------:|------:|------:|------:|------:|------:|
| 0.10 | 3.0 | 89.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0.10 | 5.0 | 196.8 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0.30 | 3.0 | 398.6 | 1.0 | 0.0 | 0.0 | 0.0 |
| 0.30 | 5.0 | 670.3 | 15.6 | 0.0 | 0.0 | 0.0 |
| 0.50 | 3.0 | 670.3 | 72.7 | 0.0 | 0.0 | 0.0 |
| 0.50 | 5.0 | 922.9 | 133.0 | 0.0 | 0.0 | 0.0 |
| 1.00 | 3.0 | 962.6 | 345.5 | 58.8 | 57.0 | 0.0 |
| 1.00 | 5.0 | 1000.0 | 364.1 | 65.0 | 64.0 | 0.0 |

Observations:

- **L6 never fires** at any tested setting. Its only input is L5 →
  weak; the L6 → L4 feedback is therefore inert. This is a structural
  consequence of the 7-edge connectivity.
- **L4 saturates** at ~1000 Hz for `w_exc ≥ 1.0`, meaning every neuron
  fires every step. This is unphysiological under any biological time
  unit interpretation.
- The "asynchronous irregular" regime that Potjans demonstrates does
  not appear at any combination tested. The closest qualitative match
  is `w_exc = 0.5, drive = 3.0`, but L23 still fires at 73 Hz vs the
  paper's 0.86 Hz.

### 5.3 Comparison to Potjans Table 4

| Population | Potjans (Hz) | This code best case (Hz) | Ratio |
|------------|--------------:|--------------------------:|------:|
| L23E | 0.86 | 1.0 (`w=0.3, thal=3`) | 1.2× |
| L23I | 2.94 | 0 (always) | 0 / not reproduced |
| L4E | 4.45 | 89 (`w=0.1, thal=3`, lowest) | 20× |
| L5E | 7.59 | 0 to 64 | 0 → 8× depending on `w_exc` |
| L6E | 1.10 | 0 (always) | 0 / not reproduced |

L4I, L5I, L6I have no analogue in this implementation.

### 5.4 Performance

`run(steps=1000, n_per_layer=20)` ≈ **39 ms** on the workstation
(Intel i5-11600K). Linear in `steps × n_per_layer²` because each layer
update is a dense `n × n` matvec. Larger `n_per_layer` quickly becomes
the bottleneck:

| `n_per_layer` | 1000-step wall (extrapolated O(n²)) |
|--------------:|------------------------------------:|
| 20 | 39 ms (measured) |
| 100 | ~1 s |
| 1 000 | ~100 s |

For Potjans' 21 915-neuron L4E alone, the current implementation would
take ~10⁵ s per second of simulated time — unusable. Restoring fidelity
would require either Rust compute or sparse matrices (or both).

---

## 6. Pipeline wiring

| Surface | How it's wired | Verifier |
|---------|---------------|----------|
| `from sc_neurocore.network import CorticalColumn` | `network/__init__.py:27` re-exports | `tests/test_cortical_column.py::test_step_output_keys` |
| `col.step(thal)` | one timestep advance + spike emit per layer | 10 of 11 tests in `test_cortical_column.py` |
| `col.run(thal, steps)` | repeated `step()` with stacked output | `test_run_output_shapes` |
| `col.reset()` | zeros all `v_*` arrays | `test_reset` |

Note: `CorticalColumn` is **not** a `Population` — it does not register
into a `Network` via `Network.add(col)` (`Network.add` raises
`TypeError` for unknown classes, `network.py:78`). It is a standalone
research toy. To use it inside a wider simulation, drive it from outside
in a manual loop.

---

## 7. Tests

```bash
PYTHONPATH=src python3 -m pytest tests/test_cortical_column.py \
                                 tests/test_cortical_column_dynamics.py -q
# 21 passed (verified 2026-04-17)
```

What the tests cover:

- Output shape and key set
- Spike values are binary
- Strong thalamic drive produces L4 spikes (passes because L4 always
  fires under any input above zero)
- Activity propagates from L4 → L23 → L5 with `w_exc=1.0` (only this
  test forces the strong-weight regime)
- Inhibition reduces L23E spikes (with strong inhibition vs weak)
- Reset clears state
- Same seed → same output (determinism)
- Run output shape `(steps, n_per_layer)`

What the tests **do not** verify:

- **Fidelity to Potjans & Diesmann** — no test asserts published firing
  rates, no test asserts the Binzegger probability matrix, no test
  asserts the 8-population structure. The current 5-population sketch
  passes the test suite even though it misses 3 populations and 57
  connections.
- **Sparseness** — no test asserts an "asynchronous irregular" CV(ISI)
  > 1 or a target rate distribution.
- **Conduction delays / PSP kernel** — none implemented, none tested.

---

## 8. Audit (7-point checklist)

| # | Dimension | Status | Detail |
|---|-----------|--------|--------|
| 1 | Pipeline wiring | ✅ PASS | re-exported, used as standalone class |
| 2 | Multi-angle tests | ⚠️ WARN | 21 tests pass, but none verify cited-paper fidelity (§7) |
| 3 | Rust path | ❌ FAIL | pure Python; for 1000-neuron columns the `n²` matvec dominates (§5.4); Rustification deferred to task #13 |
| 4 | Benchmarks | ✅ PASS | §5.4 measured here; gap to paper-scale documented |
| 5 | Performance docs | ✅ PASS | §5 |
| 6 | Documentation page | ✅ PASS | this page |
| 7 | Rules followed | ❌ FAIL | **Cited-publication fidelity violation** — module docstring cites Potjans & Diesmann 2014 but implementation reproduces ~30 % of their specification (§4). Task #10 tracks the restoration. SPDX header ✅ otherwise |

Net: **2 WARN, 2 FAIL.** This is the most violated module of the P0
sweep so far. The two FAILs are interlinked: until the model matches
Potjans (FAIL #7), benchmarking it as a "Potjans implementation"
(implicit in FAIL #3 / §5.4) is moot.

---

## 9. Known issues (for the implementation, not the doc)

These are the issues this doc surfaces. None are fixed here; all are
tracked as follow-ups.

1. **Cited-paper fidelity gap** — see §4. Either implement the full
   Potjans 2014 spec (8 populations, Binzegger matrix, PSP kernel,
   delays, Poisson background) **or** remove the Potjans citation from
   the module docstring and re-name the class to something less
   load-bearing (e.g. `MinimalCorticalSketch`). Tracked: task #10.
2. **L6 silent regime** — L6 has only one input (L5) and contributes
   only one weak feedback (to L4 at 0.5×`w_exc`). Empirically L6 never
   fires at any setting (§5.2). Either add direct external drive to L6
   (Potjans does) or document it as a known limitation.
3. **L23I silent at low weights** — at biologically plausible-looking
   weights (`w_exc ≤ 0.5`) the L23 inhibitory population never fires;
   inhibition is therefore inert. The chain L23E → L23I → L23E that
   should provide local inhibitory feedback is broken.
4. **Default parameters produce no propagation** (§5.1). The class
   ships with parameters that fail its own intended use. Either change
   defaults to a working regime (e.g. `w_exc = 1.0, threshold = 0.5`
   per the test `test_activity_propagates_to_l5`) or document that the
   user must tune.
5. **Dimensionless units** — `threshold = 1.0`, `tau = 10`. Users
   converting from biological models will silently mis-scale. Either
   adopt mV / ms units explicitly or rename parameters
   (`threshold_units = "arbitrary"`).

---

## 10. References

Cited by the module docstring (these are the papers the implementation
claims to follow):

- Douglas R. J., Martin K. A. C. "Neuronal circuits of the neocortex."
  *Annu Rev Neurosci* 27:419-451 (2004).
- Potjans T. C., Diesmann M. "The cell-type specific cortical microcircuit:
  relating structure and activity in a full-scale spiking network model."
  *Cerebral Cortex* 24(3):785-806 (2014). DOI: 10.1093/cercor/bhs358.

Background on the canonical column structure:

- Mountcastle V. B. "The columnar organization of the neocortex."
  *Brain* 120:701-722 (1997).
- Binzegger T., Douglas R. J., Martin K. A. C. "A quantitative map of
  the circuit of cat primary visual cortex." *J Neurosci*
  24(39):8441-8453 (2004). The connectivity matrix Potjans normalised.
- Hubel D. H., Wiesel T. N. "Receptive fields, binocular interaction
  and functional architecture in the cat's visual cortex." *J Physiol*
  160:106-154 (1962). Original columnar evidence.

Internal:

- Network simulation engine: [`api/network.md`](network.md)
- Other simplified-circuit page: planned `api/gamma_oscillation.md`
  (PINGCircuit has the same fidelity-gap pattern)

---

## 11. Auto-rendered API

::: sc_neurocore.network.cortical_column
    options:
      show_root_heading: true
      show_source: true
      members:
        - CorticalColumn

# Gamma Oscillation Circuit (PINGCircuit)

**Module:** `sc_neurocore.network.gamma_oscillation`
**Source:** `src/sc_neurocore/network/gamma_oscillation.py` — 120 LOC,
single `PINGCircuit` dataclass
**Status (v3.14.0):** simplified mean-field-rate sketch of pyramidal-
interneuron gamma. The class **cites** Whittington et al. 1995 and
Börgers & Kopell 2003 in its module docstring, but the implementation
**does not reproduce** either paper's mechanism. This page documents the
cited specifications, the actual implementation, the gap between them,
the empirical dynamics of the current code, and a non-determinism bug
that makes the same parameters produce different output on every call.
A fidelity-restoration follow-up is tracked as task #11.

> **Honesty notice.** Read [§4 Gap Analysis](#4-gap-analysis-vs-cited-papers),
> [§5 Empirical Dynamics](#5-empirical-dynamics-of-the-current-implementation),
> and [§6 Non-Determinism Bug](#6-non-determinism-bug) before relying on
> this code for anything that claims to model gamma oscillations. The
> current implementation produces gamma-band peaks only inside a narrow
> parameter window, fires inhibitory neurons at unphysiological 700+ Hz
> rates, and gives different results on identical inputs because two
> internal RNG calls are not seeded.

---

## 1. What the cited papers specify

### 1.1 Whittington, Traub, Jefferys 1995

*Nature* 373:612-615 (1995) — *Synchronized oscillations in interneuron
networks driven by metabotropic glutamate receptor activation*. The
**original PING / ING** observation:

- Hippocampal slice, pharmacologically activated metabotropic glutamate
  receptors on **inhibitory interneurons** drive 30–80 Hz population
  oscillation.
- Mechanism: tonic depolarisation of the inhibitory population →
  network-wide synchronous bursts of the IPSC → coherent gamma rhythm.
- The interneuron-only variant ("ING") establishes the principle that
  the **GABAₐ time constant** sets the period:
  `T ≈ τ_GABA × ln(g_synap / g_threshold)` ≈ 25 ms ⇒ ~40 Hz.
- The full PING extends ING with pyramidal cells whose firing both
  drives the interneurons and is paced by them.

The paper specifies conductance-based Hodgkin-Huxley-style cells with
explicit GABAₐ and AMPA kinetics. Without conductance-based synapses
and the τ_GABA-driven IPSC, you do not get the Whittington gamma.

### 1.2 Börgers & Kopell 2003

*Neural Computation* 15(3):509-538 (2003) — *Synchronization in Networks
of Excitatory and Inhibitory Neurons with Sparse, Random Connectivity*.
The **theoretical analysis** of PING:

- N_E excitatory + N_I inhibitory neurons, sparse Erdős–Rényi
  connectivity (each E cell connected to a random ~25 of N_I, each I
  cell similarly).
- Conductance-based reduced spiking model (theta-neuron / Wang-Buzsáki),
  not a rate model.
- Synaptic kinetics: α-function or biexponential, separate τ_AMPA and
  τ_GABA.
- Oscillation **frequency emerges** from `τ_GABA` and the coupling
  strengths via the Börgers–Kopell formula; it is not hand-tuned by
  ratio of `tau_e / tau_i`.
- Robustness: gamma persists over a wide parameter window (Figure 4 of
  the paper) — the network is **structurally stable**, not fragile.

Reproducing Börgers & Kopell means having the conductance-based
dynamics, the sparse random connectivity, and the τ_GABA-set frequency
that does not jump out of band when the drive is doubled.

---

## 2. What this implementation has

`PINGCircuit` is a dataclass:

```python
@dataclass
class PINGCircuit:
    n_excitatory: int = 80
    n_inhibitory: int = 20
    tau_e: float = 20.0          # ms
    tau_i: float = 10.0          # ms (fast-spiking)
    w_ei: float = 0.5            # E→I weight
    w_ie: float = 0.8            # I→E weight (inhibitory)
    w_ee: float = 0.1            # E→E recurrent
    threshold: float = 1.0
    reset: float = 0.0
    v_e: np.ndarray = field(default=None)  # type: ignore[arg-type]
    v_i: np.ndarray = field(default=None)
```

### 2.1 Membrane and synaptic mechanism

The `step` method (`gamma_oscillation.py:75`) updates the populations as:

```python
rate_e = np.mean(self.v_e > self.threshold * 0.8)
rate_i = np.mean(self.v_i > self.threshold * 0.8)

i_e = drive + w_ee * rate_e * n_excitatory - w_ie * rate_i * n_inhibitory
dv_e = (-self.v_e + np.maximum(i_e, 0.0)) * (dt / tau_e)
dv_e += np.random.normal(0, 0.05, n_excitatory) * np.sqrt(dt)
self.v_e += dv_e

i_i = w_ei * rate_e * n_excitatory
dv_i = (-self.v_i + np.maximum(i_i, 0.0)) * (dt / tau_i)
dv_i += np.random.normal(0, 0.05, n_inhibitory) * np.sqrt(dt)
self.v_i += dv_i
```

Three things to notice:

1. **There are no synapses.** Each population sees a scalar derived from
   the *fraction of the other population* whose voltage is above
   `0.8 × threshold` — `np.mean(v > thresh*0.8)`. Multiplied by the
   population size and a scalar weight. This is a **mean-field rate
   approximation**, not a spiking model.
2. **There is no real connectivity.** Every E cell sees the same drive
   from "the I population"; there is no sparse random matrix between
   them. Whittington/Börgers depend on the connectivity for phase locking.
3. **`np.random.normal(...)` is the global NumPy RNG** — there is no
   per-instance seed for the noise. Two `PINGCircuit` instances
   constructed with identical parameters will produce different output.
   See §6.

### 2.2 No conductance, no GABAₐ kinetics

The model has no `tau_AMPA`, no `tau_GABA`, no reversal potentials, no
α-functions. The `tau_e` and `tau_i` parameters are membrane time
constants, not synaptic kinetics. The Whittington/Börgers gamma frequency
formula (set by τ_GABA) cannot apply because τ_GABA does not exist in
this code.

### 2.3 No initial-state seed

`__post_init__` (`gamma_oscillation.py:69`) initialises voltages with
`np.random.uniform(0, 0.5, ...)` — again the global RNG. Even before
the first step, two instances differ.

```python
def __post_init__(self) -> None:
    if self.v_e is None:
        self.v_e = np.random.uniform(0, 0.5, self.n_excitatory)
    if self.v_i is None:
        self.v_i = np.random.uniform(0, 0.5, self.n_inhibitory)
```

`reset_state()` (`gamma_oscillation.py:118`) re-randomises voltages with
the same unseeded RNG.

---

## 3. Public surface

```python
@dataclass
class PINGCircuit:
    n_excitatory: int = 80
    n_inhibitory: int = 20
    tau_e: float = 20.0
    tau_i: float = 10.0
    w_ei: float = 0.5
    w_ie: float = 0.8
    w_ee: float = 0.1
    threshold: float = 1.0
    reset: float = 0.0

    def step(self, drive: float = 5.0, dt: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """One timestep. Returns (spikes_e[bool n_e], spikes_i[bool n_i])."""

    def reset_state(self) -> None:
        """Re-randomise membrane voltages (using the global numpy RNG)."""
```

There is no `run(...)` convenience method; users must call `step` in
their own loop and accumulate spike vectors.

The constructor accepts `v_e` and `v_i` as optional ndarrays (typed
`np.ndarray = field(default=None)` with `# type: ignore[arg-type]` —
the rationale for the type ignore is undocumented; mirrors `cli.py:298`).

---

## 4. Gap analysis vs cited papers

| Aspect | Whittington 1995 | Börgers & Kopell 2003 | This code | Gap |
|--------|-------------------|------------------------|-----------|-----|
| Cell model | Hodgkin–Huxley conductance-based | reduced spiking (theta / Wang-Buzsáki) | linearised mean-field rate | no spiking dynamics |
| Synapse model | AMPA + GABAₐ conductance with kinetics | α-function / biexponential | scalar `population_fraction × weight` | no synapse |
| Connectivity | full network of cells | sparse Erdős–Rényi (~25 in/cell) | none (mean-field) | absent |
| GABA time constant `τ_GABA` | sets the gamma period | sets the gamma period | not present | mechanism missing |
| Drive | tonic depolarisation via mGluR | tonic input | scalar `drive` argument to `step` | qualitatively similar |
| Frequency emergence | from `τ_GABA` and synaptic strength | from `τ_GABA`, K_E/I, drive | from `tau_e` / `tau_i` ratio + handweighted | wrong knob |
| Robustness | gamma over 30–80 Hz physiological window | gamma robust over wide drive range | gamma exists in a narrow drive window only (§5.2) | fragile |
| RNG | per-simulation seeding | per-simulation seeding | unseeded `np.random.normal/uniform` | non-deterministic (§6) |

**Net:** the implementation is a mean-field-rate model that uses
population-fraction-above-threshold as a proxy for "rate × scalar
weight". It does not implement the Whittington τ_GABA mechanism nor the
Börgers–Kopell sparse-network synchronisation. Removing the citations
or restoring the spiking model is tracked as task #11.

---

## 5. Empirical dynamics of the current implementation

Direct measurements on this workstation, not extrapolations.

### 5.1 Default parameters

```python
ping = PINGCircuit(n_excitatory=80, n_inhibitory=20)
for t in range(5000):                # 5000 × dt=0.1ms = 500 ms
    se, si = ping.step(drive=5.0, dt=0.1)
```

Wall: 186.6 ms. Total spikes:

| Population | Total spikes / 5000 steps | Mean rate (Hz, dt=0.1 ms) |
|------------|--------------------------:|----------------------------:|
| E (n=80) | 5 768 | **144.2** |
| I (n=20) | 7 553 | **755.3** |

The inhibitory rate of 755 Hz is unphysiological — even the fastest PV+
basket cells in cortex do not exceed ~300 Hz, and the absence of a
refractory period in this model permits arbitrarily high rates.

FFT of the E-population spike-count time series (drop first 1000 steps
as transient, search 5–200 Hz band):

| Quantity | Value |
|----------|-------|
| Dominant E-pop frequency | **145.0 Hz** |
| In Whittington/Börgers gamma band (30–80 Hz)? | **No** |

### 5.2 Drive / network-size sweep

Search for a parameter setting that produces gamma-band peak.
5 drives × 2 network sizes × 5000 steps each, FFT search in 5–300 Hz:

| drive | n_e | n_i | E rate (Hz) | I rate (Hz) | peak (Hz) | in gamma? |
|------:|----:|----:|------------:|------------:|----------:|:---------:|
| 1.0 | 80 | 20 | 0.3 | 21.1 | 27.5 | No |
| 1.0 | 200 | 50 | 0.1 | 20.5 | 180.0 | No |
| **3.0** | **80** | **20** | **32.4** | **372.2** | **65.0** | **Yes** |
| 3.0 | 200 | 50 | 5.8 | 100.0 | 15.0 | No |
| 5.0 | 80 | 20 | 139.9 | 720.7 | 145.0 | No |
| 5.0 | 200 | 50 | 21.3 | 302.6 | 5.0 | No |
| 10.0 | 80 | 20 | 384.9 | 729.1 | 275.0 | No |
| 10.0 | 200 | 50 | 302.1 | 1756.6 | 292.5 | No |
| 20.0 | 80 | 20 | 850.1 | 698.7 | 155.0 | No |
| 20.0 | 200 | 50 | 768.9 | 1728.3 | 170.0 | No |

Observations:

- Only **one** of the 10 parameter combinations tested falls in the
  gamma band (drive=3.0, default network size, peak = 65 Hz).
- A 1.7× drive change (3.0 → 5.0) shifts the peak from **65 Hz to
  145 Hz** — a 2.2× frequency jump. This is the opposite of the
  Börgers–Kopell robustness (their gamma persists over orders-of-
  magnitude drive changes).
- A 2.5× network-size change (80/20 → 200/50) at the same drive shifts
  the peak from **65 Hz to 15 Hz** — a 4× frequency drop. The mean-field
  approximation does not preserve the synaptic-time-constant-set
  frequency that Whittington/Börgers depend on.
- The "successful" gamma case (drive=3, peak=65 Hz) drives the I
  population at **372 Hz** — already above physiological limits.

### 5.3 Performance

`step()` × 5000 ≈ **187 ms** at default sizes. Per-step cost is
dominated by the two `np.random.normal(...)` draws and the two
`np.maximum(i, 0.0)` operations — both `O(n)`. Larger networks scale
linearly (no `n²` matvec because there is no real connectivity).

---

## 6. Non-determinism bug

`PINGCircuit.__init__` does **not** accept a `seed` argument. The
constructor's `__post_init__` and `step` both call `np.random.uniform`
or `np.random.normal` without a per-instance RNG. Two instances built
with identical parameters produce different output on every run.

Demonstration (5 runs, identical config, FFT peak frequency reported):

| Run | Total E spikes | Peak (Hz) |
|----:|---------------:|----------:|
| 0 | 762 | 5.0 |
| 1 | 531 | 60.0 |
| 2 | 951 | 60.0 |
| 3 | 816 | 95.0 |
| 4 | 819 | 85.0 |

Total spike count varies by **78 %** between runs (531 ↔ 951). Peak
frequency varies between **5 Hz and 95 Hz**. Two of five runs land in
the gamma band; the rest do not. There is no way to reproduce a result
without globally seeding NumPy before construction.

Fix: add a `seed: int = 42` parameter to the dataclass; construct
`self._rng = np.random.default_rng(seed)` in `__post_init__`; replace
every `np.random.normal(...)` / `np.random.uniform(...)` call (in
`__post_init__`, `step`, and `reset_state`) with the corresponding
`self._rng` method. Tracked as task #22 (kept separate from task #11
because it is a one-line surface change, independent of the larger
fidelity restoration).

---

## 7. Pipeline wiring

| Surface | How it's wired | Verifier |
|---------|---------------|----------|
| `from sc_neurocore.network import PINGCircuit` | `network/__init__.py:28` re-exports | `tests/test_gamma_oscillation.py` (smoke) |
| `ping.step(drive, dt)` | one timestep | smoke test |
| `ping.reset_state()` | re-randomise voltages | smoke test |

`PINGCircuit` is **not** a `Population` — it does not register into a
`Network` via `Network.add(ping)`. It is a standalone class. Spikes are
returned per-step; the user must record them in their own loop.

---

## 8. Tests

```bash
PYTHONPATH=src python3 -m pytest tests/test_gamma_oscillation.py -q
# 5 passed (verified 2026-04-17)
```

What the tests cover (5 tests, single `TestPINGCircuit` class):

- `test_creates_default` — instance constructs with defaults.
- `test_produces_spikes` — under non-zero drive, at least one spike
  fires somewhere.
- `test_no_drive_no_spikes` — zero drive produces no spikes.
- `test_inhibition_suppresses` — increasing `w_ie` reduces E activity.
- `test_reset` — `reset_state()` returns voltages to a fresh sample.

What the tests **do not** verify:

- **Gamma frequency emergence** — no test asserts an FFT peak in
  30–80 Hz. The current implementation passes its tests despite peaking
  at 145 Hz at default parameters.
- **Spiking PING mechanism** — no test asserts conductance-based
  synapses, τ_GABA-set period, or sparse connectivity.
- **Determinism** — no test asserts that two instances with identical
  parameters produce the same output. They don't (§6), and the test
  suite would not catch any regression that made it worse.
- **Comparison to Whittington 1995 or Börgers & Kopell 2003** — no test
  references the cited papers' Figure 4 / Table results.

---

## 9. Audit (7-point checklist)

| # | Dimension | Status | Detail |
|---|-----------|--------|--------|
| 1 | Pipeline wiring | ✅ PASS | re-exported, used as standalone class |
| 2 | Multi-angle tests | ⚠️ WARN | 4 smoke tests pass; no fidelity, frequency, or determinism assertion (§8) |
| 3 | Rust path | ❌ FAIL | pure Python; no Rust path planned (mean-field arithmetic could trivially be Rustified, but the priority is fidelity restoration first) |
| 4 | Benchmarks | ✅ PASS | §5 measured here; gap to paper documented |
| 5 | Performance docs | ✅ PASS | §5.3 |
| 6 | Documentation page | ✅ PASS | this page |
| 7 | Rules followed | ❌ FAIL | **Cited-publication fidelity violation** — module docstring cites Whittington 1995 + Börgers & Kopell 2003 but implementation is a mean-field-rate model (§4). Plus **non-determinism bug** (§6) violates reproducibility. Plus undocumented `# type: ignore[arg-type]` on lines 66-67. Tasks #11 (fidelity) and #22 (non-determinism) track the fixes. SPDX header ✅ otherwise |

Net: **2 WARN, 2 FAIL.** Same shape as `cortical_column.md` audit —
fidelity violation is the headline.

---

## 10. Known issues (for the implementation, not the doc)

These are the issues this doc surfaces. None are fixed here; all are
tracked as follow-ups under task #11.

1. **Cited-paper fidelity gap** — see §4. Either implement the full
   PING with conductance-based synapses, sparse random connectivity,
   τ_GABA-set frequency (Whittington/Börgers) **or** remove the
   citations and rename the class to something less load-bearing
   (e.g. `EIRateOscillator`).
2. **Non-determinism bug** — §6. Add `seed` parameter to the dataclass,
   plumb through `np.random.default_rng(seed)`. This is independent of
   the fidelity fix and should be done first.
3. **Inhibitory population fires at unphysiological rates** — 372 Hz
   in the "successful" gamma case, 700–1700 Hz in others. The model has
   no refractory period; even nominal LIF cells should clip at
   `1 / t_ref` ≈ 250 Hz.
4. **Frequency is not robust** — 1.7× drive change shifts peak by 2.2×
   (§5.2). This is the opposite of Whittington/Börgers PING, which is
   stable over orders-of-magnitude drive changes. Caused directly by
   the mean-field approximation replacing the τ_GABA mechanism.
5. **`# type: ignore[arg-type]` on dataclass field defaults**
   (`gamma_oscillation.py:66-67`) without rationale. Mirror of
   `cli.py:298`. Either type-correctly (e.g. `Optional[np.ndarray]`)
   or annotate the reason.
6. **No `run(...)` convenience method** — users must build their own
   spike-recording loop. `cortical_column.py` has one; `PINGCircuit`
   does not. Add for consistency.

---

## 11. References

Cited by the module docstring:

- Whittington M. A., Traub R. D., Jefferys J. G. R. "Synchronized
  oscillations in interneuron networks driven by metabotropic
  glutamate receptor activation." *Nature* 373:612-615 (1995).
- Börgers C., Kopell N. "Synchronization in Networks of Excitatory and
  Inhibitory Neurons with Sparse, Random Connectivity." *Neural
  Computation* 15(3):509-538 (2003).

Background on PING / cortical gamma:

- Buzsáki G., Wang X.-J. "Mechanisms of Gamma Oscillations." *Annu Rev
  Neurosci* 35:203-225 (2012). The textbook review.
- Tiesinga P., Sejnowski T. J. "Cortical Enlightenment: Are Attentional
  Gamma Oscillations Driven by ING or PING?" *Neuron* 63(6):727-732
  (2009).
- Wang X.-J., Buzsáki G. "Gamma Oscillation by Synaptic Inhibition in a
  Hippocampal Interneuronal Network Model." *J Neurosci* 16:6402-6413
  (1996). The reduced-spiking model that Börgers & Kopell adopt.

Internal:

- Network simulation engine: [`api/network.md`](network.md)
- Other simplified-circuit page: [`api/cortical_column.md`](cortical_column.md)
  (CorticalColumn has the same fidelity-gap pattern)

---

## 12. Auto-rendered API

::: sc_neurocore.network.gamma_oscillation
    options:
      show_root_heading: true
      show_source: true
      members:
        - PINGCircuit

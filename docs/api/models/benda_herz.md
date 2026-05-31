# BendaHerzNeuron

**Module:** `sc_neurocore.neurons.models.benda_herz`
**Rust:** maintained scalar safety counterpart mirrors validation and adaptation contracts
**Reference:** Benda, J. & Herz, A. V. M. (2003)
**Publication:** *A universal model for spike-frequency adaptation.* Neural Computation, 15(11), 2523–2564.
**Family:** Phenomenological spike-frequency adaptation (stochastic)
**State variables:** `a` (adaptation variable)

---

## Equations

### Instantaneous f–I curve (onset rate)

$$f_{onset}(x) = \frac{f_{max}}{1 + \exp(-\beta(x - I_{half}))}$$

### Effective firing rate (adapted)

$$f = f_{onset}(I - A)$$

The adaptation variable A shifts the f–I curve rightward: higher A means
more current is needed to produce the same rate. This is the core SFA
mechanism.

### Adaptation dynamics

$$\frac{dA}{dt} = -\frac{A}{\tau_a} + \delta_a \cdot f$$

Between spikes: A decays toward 0 with time constant τ_a = 100 ms.
During firing: A accumulates at rate δ_a × f (higher rate → faster
adaptation build-up). This creates the negative feedback loop: firing
increases A → A reduces effective drive → rate decreases.

### Stochastic spike generation

The continuous rate is integrated over the same RK4 stages used for the
adaptation state and converted to a bounded Bernoulli probability:

$$p = 1 - \exp\left(-\frac{dt}{1000}\,\bar f_{RK4}\right)$$

where

$$\bar f_{RK4} = \frac{f_1 + 2f_2 + 2f_3 + f_4}{6}$$

with each stage rate evaluated at the corresponding adaptation RK4 stage.
This keeps the rate-to-spike conversion bounded without clipping while the
adaptation ODE is integrated as a candidate-first fourth-order step.

$$\text{spike} = \begin{cases} 1 & \text{with probability } p \\ 0 & \text{otherwise} \end{cases}$$

### Implementation

```python
def step(self, current: float) -> int:
    next_a, p = self._rk4_candidate(current)
    self.a = next_a
    return 1 if self._rng.random() < p else 0
```

The candidate helper computes all RK4 adaptation stages and the exponential
hazard probability before mutating state. Python uses per-instance
`np.random.Generator`, optionally seeded by `seed`; scalar accelerator
surfaces use deterministic threshold fields where a full RNG service is not
present.

---

## Validation Contract

- `a`, `f_max`, `beta`, `i_half`, `tau_a`, `delta_a`, `dt`, and runtime `current` must be finite.
- `seed` must be `None` or a uint64-compatible integer; boolean seeds are rejected.
- `a` and `delta_a` must be non-negative; `f_max`, `beta`, `tau_a`, and `dt` must be strictly positive.
- The onset rate uses an overflow-stable logistic form and remains bounded by `[0, f_max]`.
- Each step computes all RK4 adaptation stages, an exponential hazard probability, and the candidate adaptation state before mutation.
- Probability must remain finite and within `[0, 1]`; every RK4 stage and the final candidate adaptation must remain finite and non-negative.
- Python raises `ValueError` on invalid candidates; Rust, Go, Julia, and Mojo fail closed without mutating adaptation or reporting a spike.
- `reset()` clears only dynamic adaptation `a` and preserves physical parameters.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `a` | 0.0 | a.u. | Adaptation variable (SFA accumulator) |
| `f_max` | 200.0 | Hz | Maximum firing rate |
| `beta` | 0.1 | a.u.⁻¹ | Sigmoid slope of f–I curve |
| `i_half` | 5.0 | a.u. | Half-activation current |
| `tau_a` | 100.0 | ms | Adaptation time constant |
| `delta_a` | 0.5 | a.u./Hz | Adaptation increment per Hz of firing |
| `dt` | 1.0 | ms | Integration timestep |

### Key parameter relationships

- **f_onset(I_half) = f_max/2 = 100 Hz:** At the half-activation current,
  the onset rate is half-maximum.
- **beta = 0.1:** Moderate sigmoid slope. The f–I curve transitions from
  near-zero to f_max over ~40 units of current (10%–90% range ≈ 2.2/β×2).
- **τ_a = 100 ms:** Adaptation operates on the ~100 ms timescale —
  matching the spike-frequency adaptation observed in cortical pyramidal
  neurons (Benda & Herz 2003, Fig. 3).
- **delta_a = 0.5:** Each Hz of firing rate adds 0.5 units/s to A.
  At f=100 Hz: dA/dt = 0.5×100 = 50 units/s (fast build-up).

---

## Analytical Properties

### Adaptation steady state

Setting dA/dt = 0:
$$A_{ss} = \tau_a \cdot \delta_a \cdot f_{ss} = 100 \times 0.5 \times f_{ss} = 50 f_{ss}$$

The adapted rate $f_{ss}$ satisfies:
$$f_{ss} = f_{onset}(I - A_{ss}) = f_{onset}(I - 50 f_{ss})$$

This is a self-consistency equation — the adapted rate depends on the
adaptation variable, which depends on the rate. For small f_ss:
$$f_{ss} \approx \frac{f_{onset}(I)}{1 + 50 \cdot \delta_a \cdot f'_{onset}(I)}$$

where $f'_{onset}$ is the derivative of the onset curve. Adaptation
reduces the rate by a factor proportional to the local slope.

### Adaptation onset time

Starting from A=0 with constant input I:
- A builds up exponentially with effective time constant:
$$\tau_{eff} \approx \frac{\tau_a}{1 + \tau_a \cdot \delta_a \cdot f'_{onset}}$$

For high drive: τ_eff < τ_a (adaptation is faster than its bare decay
because the rate-dependent term accelerates it).

### Stochastic spike statistics

The spike train is a **non-homogeneous Poisson process** with
time-varying rate f(t). Properties:
- **Mean ISI:** ≈ 1000/f(t) ms (instantaneous)
- **CV(ISI) ≈ 1:** Poisson-like (exponential ISI distribution)
- **Fano factor → 1:** Spike count variance equals mean (for stationary rate)

### f–I curve shape

| Current I | f_onset (Hz) | p (per step) | Expected spikes/s |
|-----------|-------------|-----|-------------------|
| 0 | ~0.3 | 0.0003 | ~0.3 |
| 5 (= I_half) | 100 | 0.1 | 100 |
| 10 | ~197 | 0.197 | ~197 |
| 50 | ~200 | 0.2 | ~200 (saturated) |

With adaptation (A > 0), the effective current is I − A, shifting the
entire table rightward.

### Rate-to-spike conversion

The Bernoulli sampling converts the continuous rate to a binary spike
train. The conversion is exact in expectation:
$$E[\text{spikes per second}] = f \cdot 1000/dt$$

But individual realisations are noisy. For reliable rate estimation,
many steps are needed (~1000+ for 10% accuracy).

---

## Behaviour

### Spike-frequency adaptation (SFA)

The core feature of the Benda-Herz model:

1. **Onset response:** Input arrives → A=0 → full f_onset → high rate
2. **Adaptation:** High rate → A builds up → effective drive decreases
3. **Adapted response:** Rate settles to f_ss < f_onset (reduced)
4. **Recovery:** Input removed → A decays → f_onset potential restored

The adaptation time course is exponential with τ_a = 100 ms.

### Stochastic nature

Unlike deterministic models (LIF, HH, AdEx), the BendaHerz neuron
produces **different spike trains on every run.** Two instances with
identical parameters and identical input will produce different spike
times — only the underlying rate is the same.

This stochasticity is a feature, not a bug: it models the trial-to-trial
variability observed in cortical neurons. The `_rng` per-instance
Generator ensures reproducibility when seeded.

### Dual nature: rate model + spiking output

The model is conceptually a **rate model** (computes f in Hz) with a
**stochastic spike output** (Bernoulli sampling). This bridges two
paradigms:
- Rate models: analytical, fast, but no spike timing
- Spiking models: spike timing, but computationally expensive

The BendaHerz model combines analytical rate computation with biologically
realistic stochastic spike output.

### Adaptation reduces firing rate

Verified by test: after 1000 steps at I=30, A > 0 (adaptation accumulated).
The adaptation variable shifts the f–I curve rightward, requiring more
input current to achieve the same rate.

### Adaptation variable A accumulates

Verified by test: A increases from 0 under sustained drive. The
accumulation rate is delta_a × f — proportional to the instantaneous
firing rate.

---

## Benda & Herz 2003 Context

### Phenomenological vs biophysical SFA

Biophysical SFA models (AdEx, HH with Ca²⁺-activated K⁺) derive
adaptation from ion channel dynamics. The Benda-Herz model is
**phenomenological:** it captures the input-output relationship of SFA
(rightward f–I shift) without modelling the underlying biophysics.

Advantages of the phenomenological approach:
- **Faster:** 1 exp() per step (sigmoid) vs multiple ion channels
- **Fewer parameters:** 7 vs 10+ for biophysical models
- **Analytically tractable:** Steady-state rate can be computed
- **Experimentally grounded:** Parameters map directly to measurable
  quantities (onset f–I curve, adaptation time constant)

### Subtractive vs divisive adaptation

Benda & Herz (2003) showed that SFA can be classified as:
- **Subtractive:** A subtracts from the input (this model: f(I − A))
  → rightward shift of f–I curve
- **Divisive:** A divides the gain → compression of f–I curve

This model implements **subtractive adaptation.** The distinction matters
for neural coding: subtractive adaptation shifts the operating point,
while divisive adaptation changes the sensitivity.

---

## Comparison with Related Models

| Property | BendaHerz | AdEx | SRM0 | StochasticIF |
|----------|----------|------|------|-------------|
| Adaptation | Subtractive (A) | Current (w) | Kernel (η) | None |
| Stochastic | Yes (Poisson) | No | No | Yes (OU noise) |
| f–I computation | Explicit (sigmoid) | Implicit (from ODE) | Implicit | Implicit |
| State vars | 1 (A) | 2 (V, w) | 1 (V) + η | 1 (V) |
| Rate model | Yes (internal) | No | No | No |
| Spike output | Bernoulli | Deterministic | Deterministic | Noisy threshold |
| Speed | ~500K steps/s | ~500K steps/s | ~400K steps/s | ~500K steps/s |

---

## Polyglot Surfaces

Python, Go, Julia, Mojo, and the Rust safety surface now implement the same
Benda-Herz contract: overflow-stable onset rate, candidate-first RK4
adaptation integration, exponential hazard probability, fail-closed invalid
input handling, and reset-preserving parameter semantics.

Fresh evidence for this slice:

- `PYTHONPATH=src .venv/bin/python -m pytest tests/test_model_benda_herz.py -q` → `53 passed`.
- `GO111MODULE=off go test ./src/sc_neurocore/accel/go/services -run 'BendaHerz' -bench 'BenchmarkBendaHerzStep' -benchmem` → Benda-Herz tests passed; `235.7 ns/op`, `0 B/op`, `0 allocs/op`.
- `rustc --test src/sc_neurocore/accel/rust/safety/benda_herz.rs ...` → `10 passed`.
- `benchmarks/results/local_i5_11600k_python_2026-06-01_benda_herz.json` records the refreshed Python RK4 hazard benchmark.

---

## Pipeline Verification (End-to-End, Measured 2026-06-01)

Focused module verification passed with the repository virtual environment:

```text
PYTHONPATH=src .venv/bin/python -m pytest tests/test_model_benda_herz.py -q
53 passed in 20.07s
```

The module-specific suite covers construction, finite-domain validation,
overflow-stable onset rates, RK4 candidate parity against an independent
reference, exponential hazard probability bounds, seeded stochastic
reproducibility, fail-closed state preservation, reset semantics, population
wiring, projection wiring, network execution, and spike-statistics analysis.

### Pipeline stages verified

| Stage | Status | Notes |
|-------|--------|-------|
| Import + construction | PASS | Validates finite physical parameters and optional seed |
| `step()` → int `{0,1}` | PASS | Bernoulli output from bounded exponential hazard |
| RK4 adaptation candidate | PASS | Independent reference parity at `1e-14` Python tolerance |
| Adaptation accumulates | PASS | `a` increases under sustained drive |
| f-onset sigmoid | PASS | Monotonic, bounded `[0, f_max]` |
| Invalid runtime input | PASS | Raises before mutating Python state |
| Reset | PASS | Clears only `a` |
| Population/network wiring | PASS | `Population`, `Projection`, `PoissonInput`, `Network`, `SpikeMonitor` |
| Analysis wiring | PASS | `firing_rate` and `spike_count` consume generated spike trains |

Go service verification and benchmark:

```text
GO111MODULE=off go test ./src/sc_neurocore/accel/go/services -run 'BendaHerz' -bench 'BenchmarkBendaHerzStep' -benchmem
BenchmarkBendaHerzStep-12  4556997  235.7 ns/op  0 B/op  0 allocs/op
PASS
```

Rust safety verification:

```text
rustc --test src/sc_neurocore/accel/rust/safety/benda_herz.rs -o /tmp/sc_neurocore_benda_herz_safety_test && /tmp/sc_neurocore_benda_herz_safety_test
10 passed
```

## Numerical Considerations

- **1 exp() per step:** The f_onset sigmoid is the only transcendental.
- **1 random number per step:** _rng.random() from numpy Generator.
- **A not clipped:** Can grow without bound under sustained high drive.
  In practice, the self-consistency A_ss = 50×f_ss limits A.
- **p clipped to [0, 1]:** min(p, 1.0) prevents probability > 1.
  At f_max=200Hz and dt=1ms: p_max = 0.2 (safely below 1).
- **Per-instance RNG:** np.random.default_rng() — each neuron has its
  own Generator. Seeding at population level requires external control.

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/benda_herz.py`.
- **One state variable:** `a` (adaptation).
- **`__post_init__`:** validates finite parameters and creates a per-instance RNG via `np.random.default_rng(seed)`.
- **Private helpers:** `_f_onset()` computes the stable sigmoid; `_rk4_candidate()` computes the adaptation candidate and exponential hazard probability.
- **Dataclass:** Uses `@dataclass` with `field(init=False)` for RNG state.

---

## Performance

| Metric | Python | Notes |
|--------|--------|-------|
| Isolation | ~500K steps/s | 1 exp + 1 random per step |
| Network (20n, 2s) | ~350K neuron-steps/s | Measured |

Fast model — single sigmoid evaluation + single random number per step.

---

## Test Coverage

### Python Tests

**File:** `tests/test_model_benda_herz.py`

| Category | What is verified |
|----------|------------------|
| Isolation | Construction, binary spike result, sustained-drive spikes, reset |
| RK4 and probability | Independent RK4 candidate parity, RK4 commit semantics, exponential hazard bounds |
| Stochastic reproducibility | Optional seed produces reproducible spike sequences |
| Validation | Parameter, seed, current, stage, probability, and candidate-state fail-closed boundaries |
| Pipeline | Population, projection, network, monitor, firing-rate and spike-count analysis |

### Go Tests and Benchmark

**File:** `src/sc_neurocore/accel/go/services/benda_herz_test.go`

| Test | What is verified |
|------|------------------|
| `TestBendaHerzRK4CandidateMatchesReference` | Go RK4 candidate and hazard match independent reference |
| `TestBendaHerzStepCommitsRK4Candidate` | Step commits the RK4 candidate |
| `TestBendaHerzInvalidRuntimeInputPreservesState` | Invalid current preserves adaptation |
| `TestBendaHerzInvalidCandidatePreservesState` | Invalid RK4 candidate preserves adaptation |
| `BenchmarkBendaHerzStep` | Scalar service step latency and allocations |

### Rust Safety Tests

**File:** `src/sc_neurocore/accel/rust/safety/benda_herz.rs`

The Rust safety module has 10 focused tests covering construction, RK4
candidate parity, step commit semantics, onset-rate bounds, adaptation growth,
invalid input preservation, invalid candidate preservation, and reset semantics.

## Citations

1. **Benda, J. & Herz, A. V. M.** (2003).
   A universal model for spike-frequency adaptation.
   *Neural Computation*, 15(11), 2523–2564.
   DOI: [10.1162/089976603322385063](https://doi.org/10.1162/089976603322385063)

2. **Benda, J., Longtin, A., & Maler, L.** (2005).
   Spike-frequency adaptation separates transient communication signals
   from background oscillations.
   *Journal of Neuroscience*, 25(9), 2312–2321.
   DOI: [10.1523/JNEUROSCI.4795-04.2005](https://doi.org/10.1523/JNEUROSCI.4795-04.2005)

3. **Pozzorini, C., Naud, R., Mensi, S., & Gerstner, W.** (2013).
   Temporal whitening by power-law adaptation in neocortical neurons.
   *Nature Neuroscience*, 16(7), 942–948.
   DOI: [10.1038/nn.3431](https://doi.org/10.1038/nn.3431)

4. **Ermentrout, G. B.** (1998).
   Linearization of F-I curves by adaptation.
   *Neural Computation*, 10(7), 1721–1729.
   DOI: [10.1162/089976698300017106](https://doi.org/10.1162/089976698300017106)

5. **Gerstner, W., Kistler, W. M., Naud, R., & Paninski, L.** (2014).
   *Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition.*
   Cambridge University Press. Chapter 5: Adaptation and firing patterns.

---

*SC-NeuroCore v3.14.0 — ANULUM / Fortis Studio*
*© 2020–2026 Miroslav Šotek. All rights reserved.*


## Benchmark Note

The 2026-06-01 RK4 hazard benchmark supersedes older Benda-Herz throughput
notes for the Python and Go scalar paths. Historical Criterion artefacts remain
available for provenance but predate the current integration contract.

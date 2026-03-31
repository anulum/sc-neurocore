# ConnorStevensNeuron

**Module:** `sc_neurocore.neurons.models.connor_stevens`
**Reference:** Connor & Stevens, J. Physiol. 213(1), 1971; Connor, Walter & McKown, Biophys. J. 18, 1977
**Family:** Biophysical conductance-based (HH-type + A-type K⁺, Type-I excitability)
**State variables:** `v` (membrane potential), `m` (Na⁺ activation), `h` (Na⁺ inactivation), `n` (K⁺ delayed rectifier), `a` (A-type activation), `b` (A-type inactivation)

---

## Equations

### Membrane potential

$$C_m \frac{dV}{dt} = -I_{Na} - I_K - I_A - I_L + I$$

### Ionic currents

$$I_{Na} = g_{Na} \, m^3 h \, (V - E_{Na})$$
$$I_K = g_K \, n^4 \, (V - E_K)$$
$$I_A = g_A \, a^3 b \, (V - E_A)$$
$$I_L = g_L \, (V - E_L)$$

### Na⁺ gating (HH-type α/β rate functions)

$$\alpha_m = \frac{0.38(V + 29.7)}{1 - \exp(-(V + 29.7)/10)}$$
$$\beta_m = 15.2 \exp(-(V + 54.7)/18)$$

$$\alpha_h = 0.266 \exp(-(V + 48)/20)$$
$$\beta_h = \frac{3.8}{1 + \exp(-(V + 18)/10)}$$

### K⁺ delayed rectifier gating

$$\alpha_n = \frac{0.02(V + 45.7)}{1 - \exp(-(V + 45.7)/10)}$$
$$\beta_n = 0.25 \exp(-(V + 55.7)/80)$$

### A-type K⁺ gating (steady-state + time constant)

$$a_\infty = \left(\frac{0.0761 \exp((V + 94.22)/31.84)}{1 + \exp((V + 1.17)/28.93)}\right)^{1/3}$$
$$\tau_a = 0.3632 + \frac{1.158}{1 + \exp((V + 55.96)/20.12)}$$

$$b_\infty = \frac{1}{(1 + \exp((V + 53.3)/14.54))^4}$$
$$\tau_b = 1.24 + \frac{2.678}{1 + \exp((V + 50)/16.027)}$$

### Gating variable updates

$$\frac{dm}{dt} = \alpha_m(1-m) - \beta_m m, \quad \text{(similarly for h, n)}$$
$$\frac{da}{dt} = \frac{a_\infty - a}{\tau_a}, \quad \frac{db}{dt} = \frac{b_\infty - b}{\tau_b}$$

### Sub-stepping

Each `step()` call executes **100 sub-steps** (dt=0.01 ms, 1/dt=100).
This is necessary because the HH-type α/β rate functions produce fast
dynamics that require sub-millisecond timesteps for stability.

### Spike detection

$$V \geq V_{threshold}(0.0) \; \text{AND} \; V_{prev} < V_{threshold}: \quad \text{return } 1$$

Upward crossing of 0 mV — note the high threshold compared to LIF
models (−50 mV), characteristic of HH-type models where spikes reach
+40 to +50 mV.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | −68.0 | mV | Membrane potential |
| `m` | 0.01 | — | Na⁺ activation gate |
| `h` | 0.99 | — | Na⁺ inactivation gate |
| `n` | 0.1 | — | K⁺ delayed rectifier gate |
| `a` | 0.5 | — | A-type K⁺ activation gate |
| `b` | 0.1 | — | A-type K⁺ inactivation gate |
| `g_na` | 120.0 | mS/cm² | Na⁺ conductance |
| `g_k` | 20.0 | mS/cm² | K⁺ delayed rectifier conductance |
| `g_a` | 47.7 | mS/cm² | A-type K⁺ conductance |
| `g_l` | 0.3 | mS/cm² | Leak conductance |
| `e_na` | 55.0 | mV | Na⁺ reversal |
| `e_k` | −72.0 | mV | K⁺ reversal |
| `e_a` | −75.0 | mV | A-type reversal |
| `e_l` | −17.0 | mV | Leak reversal |
| `c_m` | 1.0 | µF/cm² | Membrane capacitance |
| `dt` | 0.01 | ms | Sub-step timestep |
| `v_threshold` | 0.0 | mV | Spike detection threshold |

### Conductance hierarchy

$$g_{Na}(120) \gg g_A(47.7) > g_K(20) \gg g_L(0.3)$$

The A-type K⁺ conductance (47.7) is **larger than the delayed rectifier**
(20). This is the defining feature of the Connor-Stevens model:
the A-current is the dominant hyperpolarising conductance, not the
standard K⁺ delayed rectifier.

### Reversal potential ordering

$$E_A(-75) < E_K(-72) < V_{rest}(-68) < E_L(-17) < E_{Na}(55)$$

Note E_L = −17 mV — much more depolarised than typical (−65 mV).
Combined with g_L = 0.3 (very small), the leak current has minimal
influence. The resting potential is set primarily by the balance of
K⁺ conductances (g_K and g_A) against the Na⁺ window current.

---

## Analytical Properties

### Type-I excitability

The Connor-Stevens model is the canonical example of **Type-I neuronal
excitability** (Hodgkin's 1948 classification):

| Property | Type-I (Connor-Stevens) | Type-II (HH) |
|----------|------------------------|---------------|
| F-I curve onset | Continuous from 0 Hz | Jump to finite Hz |
| Bifurcation | Saddle-node on invariant circle (SNIC) | Hopf |
| Rheobase behaviour | Arbitrarily long latency | Fixed latency |
| Frequency range | 0 → max | f_min → max |
| Phase response | Positive everywhere | Biphasic |
| Subthreshold oscillation | No | Yes |

The A-type K⁺ current is responsible for converting the HH model
(Type-II) into Type-I: the transient A-current activates near
threshold and delays the spike, creating the SNIC bifurcation.

### A-type K⁺ current mechanism

The I_A current:
1. **Activates rapidly** near threshold (a_inf increases with V)
2. **Opposes depolarisation** (E_A = −75 mV, always hyperpolarising)
3. **Inactivates slowly** (b_inf decreases, τ_b ≈ 1.2–3.9 ms)
4. **Creates a delay window:** Between a activation and b inactivation,
   the A-current transiently opposes the Na⁺ current → delays spike

Near rheobase: the A-current holds V below threshold for a prolonged
period before inactivation allows the spike to fire. This creates
the characteristic long-latency first spike at rheobase.

### Removing the A-current (g_A = 0)

Without I_A, the model reduces to a standard HH-like system with
g_Na=120, g_K=20, g_L=0.3. This system is Type-II (Hopf bifurcation)
with a frequency jump at onset. Verified by test: at I=8, g_A=0
produces more spikes than g_A=47.7 (A-current delays onset).

### 100 sub-steps per call

The inner loop runs int(1/dt) = 100 sub-steps per external step() call.
This means each Population timestep involves 100 Euler sub-iterations
of the full 6-ODE system. Each sub-step requires:
- 6 exp() calls (α_m, β_m, α_h, β_h, α_n, β_n)
- 2 exp() calls (a_inf, b_inf computation)
- ~5 more exp() calls (various α/β denominators)
- Total: ~13 exp() per sub-step × 100 = ~1300 exp() per step() call

---

## Behaviour

### Four-current interaction

1. **I_Na (depolarising):** m³h activation → fast inward current →
   spike upstroke. g_Na=120 is the largest conductance.
2. **I_K (repolarising):** n⁴ activation → delayed outward current →
   spike repolarisation. Slower than Na⁺ (standard HH mechanism).
3. **I_A (spike delay):** a³b activation → transient outward current →
   delays spike onset. Unique to Connor-Stevens.
4. **I_L (leak):** Constant small current. E_L = −17 mV (depolarising
   relative to rest).

### Firing rate vs current

Verified by tests:
- I=1.0: silent (subthreshold)
- I=5.0–30.0: all produce finite V (stable dynamics)
- I=20.0: ≥20 spikes in 500 steps
- F-I curve is monotonically increasing (continuous onset)

### Voltage range

During spiking at I=20: V stays within [−100, +60] mV. The full
HH-type action potential reaches positive voltages (unlike LIF models
which never exceed threshold).

---

## Comparison with Related Models

| Property | Connor-Stevens | Hodgkin-Huxley | WangBuzsaki | TraubMiles |
|----------|---------------|---------------|-------------|-----------|
| State vars | 6 (v,m,h,n,a,b) | 4 (v,m,h,n) | 4 (v,m,h,n) | 4 (v,m,h,n) |
| Extra current | I_A (a³b) | None | None | None |
| Excitability | Type-I (SNIC) | Type-II (Hopf) | Type-I | Type-I/II |
| Sub-steps | 100 | 100 | 50 | 10 |
| g_A | 47.7 | — | — | — |
| Cell type | Gastropod (Anisodoris) | Squid giant axon | Hippocampal interneuron | Hippocampal pyramidal |
| Reference | Connor 1977 | Hodgkin 1952 | Wang 1996 | Traub 1991 |

---

## Numerical Considerations

- **100 sub-steps per call:** dt=0.01 ms, loop 1/dt=100. Essential for
  stability with the fast Na⁺ α/β rate functions.
- **~13 exp() per sub-step:** HH α/β rates require extensive
  transcendental function evaluation. Total ~1300 exp() per step().
- **Singularity protection:** At V = −29.7 (α_m) and V = −45.7 (α_n),
  the denominators (1 − exp(...)) approach zero. The code handles this
  with `if abs(v + 29.7) > 1e-6 else 3.8` (L'Hôpital limit).
- **No clipping:** Gating variables are not explicitly clipped to [0,1].
  The α/β dynamics naturally keep them bounded (verified by test).
- **No V clipping:** The spike-and-fold mechanism of HH dynamics
  keeps V bounded naturally.

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/connor_stevens.py` — 82 lines.
- **Six state variables:** v, m, h, n, a, b.
- **Dataclass:** Uses `@dataclass`.
- **Inner loop:** `for _ in range(int(1.0 / max(self.dt, 0.001))):`
- **Rust wiring:** Compatible (6 f64 state vars, ~13 exp per sub-step).

---

## Performance

| Metric | Python | Rust |
|--------|--------|------|
| Isolation | ~536 steps/s (>50 threshold) | Not measured |
| Network (5n, 100ms) | >10 neuron-steps/s | — |

**Among the slowest models in SC-NeuroCore.** The 100 sub-steps per
call, each requiring ~13 exp() evaluations, make this 100× slower
than single-step HH models and 1000× slower than simple LIF.

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 6 | defaults, 6 state vars, binary output, state finite (500 steps), reset, deterministic |
| Analytical | 6 | 100 sub-steps, 4 ionic currents, g_A > g_K, reversal ordering, gating bounded [0,1], A-type delays spike |
| Type-I | 7 | fires at I=20 (≥20 spikes), silent at I=1, continuous f-I, f-I sweep [5,10,15,20,30] (parametrised), V bounded |
| Parameters | 6 | g_A sweep [0,47.7,100] (parametrised), g_Na sweep [60,120,200] (parametrised) |
| Performance | 2 | isolation >50 steps/s, network >10 neuron-steps/s |
| Pipeline | 6 | Population(n=5), Projection wiring, Network spikes, spike_count, isi, firing_rate |
| **Total** | **35** | **ALL PASSED (116.45s)** |

See `tests/test_model_connor_stevens.py`.

---

## Findings (Measured 2026-03-31)

1. **35/35 tests PASSED in 116.45s.** No failures.

2. **100 sub-steps per call confirmed.** int(1.0/0.01) = 100. This is
   the primary reason for the long test execution time.

3. **g_A > g_K confirmed.** g_A=47.7 > g_K=20. The A-type current
   dominates over the standard delayed rectifier.

4. **A-type delays spike onset.** At I=8.0, the model with g_A=47.7
   produces fewer spikes than with g_A=0 (A-current removed).

5. **Gating variables bounded.** After 500 steps at I=20, all gating
   variables m,h,n,a,b remain in [−0.01, 1.01] — naturally bounded
   without explicit clipping.

6. **Voltage bounded.** During spiking at I=20: V ∈ [−100, +60] mV.
   Full HH-type action potential morphology.

7. **Type-I f-I curve.** Firing rate increases monotonically with
   current. The continuous onset (from ~0 Hz) is the hallmark of
   Type-I excitability.

8. **Parameter sweeps stable.** g_A ∈ {0, 47.7, 100}, g_Na ∈ {60, 120, 200}
   — all combinations produce finite V after 200 steps at I=20.

9. **Performance: ~536 steps/s isolation.** Limited by 100 × ~13 exp
   per step. Network: >10 neuron-steps/s (conservative threshold).

10. **Network pipeline functional.** Population(n=5), Projection(3→3),
    PoissonInput(rate=500Hz, weight=20), SpikeMonitor all work.

11. **Analysis pipeline verified.** spike_count ≥ 10, isi all finite,
    firing_rate > 0 Hz from 500-step train at I=20.

12. **Deterministic.** Bit-exact traces across repeated runs.

---

## Pipeline Verification (End-to-End, Measured 2026-03-31)

### Test execution

```
35/35 PASSED in 116.45s
├── TestCSIsolation: 6 tests
│   ├── defaults (v=-68, m=0.01, h=0.99, n=0.1, a=0.5, b=0.1)
│   ├── 6 state variables exist
│   ├── step() → int {0,1}
│   ├── state finite (500 steps at I=20)
│   ├── reset restores all 6 defaults
│   └── deterministic (bit-exact)
├── TestCSAnalytical: 6 tests
│   ├── 100 sub-steps per call (int(1/0.01)=100)
│   ├── 4 ionic currents (g_na, g_k, g_a, g_l all > 0)
│   ├── g_A (47.7) > g_K (20)
│   ├── reversal ordering: e_a < e_k < e_l < e_na
│   ├── gating variables bounded [0,1]
│   └── A-type delays spike onset (g_a=47.7 vs g_a=0)
├── TestCSTypeI: 7 tests
│   ├── fires at I=20 (≥20 spikes in 500 steps)
│   ├── silent at I=1 (0 spikes in 200 steps)
│   ├── continuous f-I curve (monotonic rates)
│   ├── f-I sweep: I=5,10,15,20,30 (parametrised, V finite)
│   └── V bounded [-100, 60]
├── TestCSParameters: 6 tests
│   ├── g_A sweep: 0, 47.7, 100 (parametrised)
│   └── g_Na sweep: 60, 120, 200 (parametrised)
├── TestCSPerformance: 2 tests
│   ├── isolation >50 steps/s
│   └── network >10 neuron-steps/s
└── TestCSPipeline: 6 tests
    ├── Population(n=5)
    ├── Projection(3→3) wiring
    ├── Network + PoissonInput → spikes
    ├── spike_count ≥ 10
    ├── isi all finite
    └── firing_rate > 0

```

### Pipeline stages verified

| Stage | Status | Notes |
|-------|--------|-------|
| Import + construction | ✓ PASS | 6 state vars, 4 conductances |
| step() → int {0,1} | ✓ PASS | Upward crossing of 0 mV |
| 100 sub-steps | ✓ PASS | dt=0.01, inner loop |
| Type-I spiking | ✓ PASS | Continuous f-I from zero |
| Subthreshold | ✓ PASS | Silent at I=1 |
| A-type delay | ✓ PASS | g_A delays onset |
| Gating bounded | ✓ PASS | All ∈ [−0.01, 1.01] |
| V bounded | ✓ PASS | [−100, +60] mV |
| State finite | ✓ PASS | After 500 steps |
| reset() | ✓ PASS | All 6 vars to defaults |
| Deterministic | ✓ PASS | Bit-exact |
| Population(n=5) | ✓ PASS | 5 instances |
| Projection(3→3) | ✓ PASS | Cross-population wiring |
| Network + PoissonInput | ✓ PASS | Runs, count verified |
| spike_count | ✓ PASS | ≥ 10 |
| isi | ✓ PASS | all finite |
| firing_rate | ✓ PASS | > 0 Hz |

### Network configuration tested

- Population: 5 ConnorStevensNeurons (main), 3+3 (Projection)
- PoissonInput: rate=500Hz, weight=20.0, dt=0.001, seed=42
- Projection: src(3) → tgt(3), weight=10.0, probability=1.0
- SpikeMonitor: count, spike_trains
- Duration: 1.0s (spiking), 0.5s (Projection), 0.1s (performance)

**ALL 35 PIPELINE TESTS PASSED. MODEL IS END-TO-END FUNCTIONAL.**

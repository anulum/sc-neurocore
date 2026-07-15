# YamadaNeuron

**Module:** `sc_neurocore.neurons.models.yamada`
**Reference:** Yamada, Kashimori & Kambara, Biol. Cybern. 61, 1989
**Family:** Biophysical conductance-based (3-ODE, subcritical Hopf burster)
**State variables:** `v` (membrane potential), `n` (K⁺ recovery), `q` (slow bursting variable)

---

## Equations

### Membrane potential

$$\frac{dV}{dt} = -I_{Na} - I_K - I_q - I_L + I$$

### Ionic currents

$$I_{Na} = g_{Na} \, m_\infty^3 (1-n) \, (V - E_{Na})$$
$$I_K = g_K \, n^4 \, (V - E_K)$$
$$I_q = g_q \, q \, (V - E_q)$$
$$I_L = g_L \, (V - E_L)$$

### Steady-state activation functions (Boltzmann sigmoids)

$$m_\infty(V) = \frac{1}{1 + \exp(-(V+30)/9.5)}$$
$$n_\infty(V) = \frac{1}{1 + \exp(-(V+30)/10)}$$
$$q_\infty(V) = \frac{1}{1 + \exp(-(V+50)/10)}$$

### Recovery variable (fast K⁺)

$$\frac{dn}{dt} = \frac{n_\infty(V) - n}{\tau_n(V)}$$

$$\tau_n(V) = 1 + \frac{7.5}{1 + \exp((V+40)/12)}$$

τ_n is voltage-dependent: slow at hyperpolarised potentials (τ_n ≈ 8.5 ms
at V = −80), fast at depolarised potentials (τ_n ≈ 1 ms at V = 0).

### Slow bursting variable

$$\frac{dq}{dt} = \frac{q_\infty(V) - q}{\tau_q}$$

τ_q = 300 ms (constant) — the slowest timescale in the model.

### Three timescales

| Variable | Timescale | Role |
|----------|-----------|------|
| V | ~0.05 ms (dt) | Fast spike dynamics |
| n | 1–8.5 ms (voltage-dep.) | Spike repolarisation |
| q | 300 ms (constant) | Burst modulation |

### Spike detection

Upward crossing: $V_t \geq V_{threshold}$ AND $V_{t-1} < V_{threshold}$.

### Implementation

```python
def step(self, current: float) -> int:
    v_prev = self.v
    k1 = rhs(v, n, q, current)
    k2 = rhs(v + 0.5*dt*k1.v, n + 0.5*dt*k1.n, q + 0.5*dt*k1.q, current)
    k3 = rhs(v + 0.5*dt*k2.v, n + 0.5*dt*k2.n, q + 0.5*dt*k2.q, current)
    k4 = rhs(v + dt*k3.v, n + dt*k3.n, q + dt*k3.q, current)
    candidate = state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    if any stage/candidate is non-finite or n/q leave [0, 1]:
        raise ValueError
    self.v, self.n, self.q = candidate
    return 1 if crossing else 0
```

Candidate-first RK4, single macro-step per call. Each of the four stages
evaluates the same Yamada conductance RHS, giving 16 sigmoid/exponential
evaluations per step. The state is committed only after all stages and the
accepted candidate remain finite and the two gating variables stay inside
`[0, 1]`.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | −60.0 | mV | Membrane potential |
| `n` | 0.1 | — | K⁺ recovery gate |
| `q` | 0.0 | — | Slow bursting variable |
| `g_na` | 20.0 | mS/cm² | Na⁺ conductance |
| `g_k` | 10.0 | mS/cm² | K⁺ conductance |
| `g_q` | 5.0 | mS/cm² | Slow current conductance |
| `g_l` | 0.5 | mS/cm² | Leak conductance |
| `e_na` | 60.0 | mV | Na⁺ reversal |
| `e_k` | −80.0 | mV | K⁺ reversal |
| `e_q` | −80.0 | mV | Slow current reversal |
| `e_l` | −60.0 | mV | Leak reversal (= V_rest) |
| `tau_q` | 300.0 | ms | Slow variable time constant |
| `dt` | 0.05 | ms | Integration timestep |
| `v_threshold` | −20.0 | mV | Spike detection threshold |

---

## Analytical Properties

### Subcritical Hopf bursting mechanism

The model produces **square-wave bursting** via slow modulation of a
Hopf bifurcation:

1. **Silent phase (q low):** The slow current I_q is weak → the system
   is below the Hopf bifurcation → stable rest (no spikes)
2. **Transition to active:** q drifts up (toward q_inf > 0) as V rises
   above −50 mV → eventually I_q provides enough negative feedback to
   create an oscillatory instability
3. **Active phase (q moderate):** The system is in a limit cycle →
   rapid spiking (burst)
4. **Burst termination:** During spiking, q increases further →
   I_q = g_q · q · (V − E_q) becomes substantially hyperpolarising →
   overwhelms the excitatory drive → system falls back to rest
5. **Recovery:** q decays slowly (τ_q = 300 ms) → cycle repeats

### m_inf is instantaneous (no state variable)

Like WangBuzsaki, the Na⁺ activation m is treated as instantaneous
(m_inf computed from V each step). This reduces the model from 4 to 3 ODEs.

### Inactivation via (1−n)

The Na⁺ current uses $(1-n)$ as the inactivation factor instead of a
separate h gate. Since n activates during the spike (K⁺ opens), $(1-n)$
decreases — mimicking Na⁺ inactivation. This is a standard simplification
(same as Wilson-Cowan derived models).

### Reversal potential ordering

$$E_K = E_q = -80 < E_L = -60 < V_{threshold} = -20 < E_{Na} = 60$$

Both K⁺ and the slow current share the same reversal (−80 mV), meaning
q acts as a second K⁺-like current but on a much slower timescale.

### Boltzmann midpoints

| Function | Midpoint | Slope factor |
|----------|----------|-------------|
| m_inf | −30 mV | 9.5 mV |
| n_inf | −30 mV | 10 mV |
| q_inf | −50 mV | 10 mV |

m_inf and n_inf share the same midpoint (−30 mV) — this means Na⁺
activation and K⁺ activation co-activate near the same voltage, with
the timing difference (m instantaneous, n delayed by τ_n) creating the
spike. q_inf activates at −50 mV — 20 mV below the fast gates —
meaning the slow variable becomes active in the subthreshold/perithreshold
regime, where it modulates excitability.

---

## Behaviour

### Square-wave bursting

The characteristic bursting pattern:
- **Burst:** 5–20 rapid spikes at high frequency
- **Inter-burst interval:** 100–500 ms of silence (q recovery)
- **Regular period:** Bursts repeat with consistent timing

### Burst duration controlled by g_q

- g_q small (1.0): long bursts (weak slow feedback → slow termination)
- g_q large (10.0): short bursts (substantial slow feedback → fast termination)
- g_q = 0: no bursting, continuous spiking (q has no effect)

### τ_q controls burst period

- τ_q = 100 ms: fast burst cycling (short inter-burst interval)
- τ_q = 300 ms: moderate (default)
- τ_q = 1000 ms: slow burst cycling (long inter-burst interval)

### Input affects burst frequency

Higher current → shorter inter-burst intervals and longer bursts.
The f-I curve for mean firing rate (averaged over bursts) is monotonic.

---

## Comparison with Related Models

| Property | Yamada | HindmarshRose | Butera | ChayKeizer |
|----------|-------|---------------|--------|-----------|
| ODEs | 3 | 3 | 3 | 3 |
| Bursting | Square-wave (Hopf) | Square-wave | Parabolic | Square-wave |
| Slow var | q (τ=300ms) | z (r=0.001) | h (τ_h) | Ca²⁺ |
| Biophysical | Semi (Boltzmann) | Polynomial | HH-like | Ion channel |
| Currents | Na, K, q, L | None explicit | Na, K, NaP, L | Na, K, Ca, K-Ca |
| m_inf | Yes (instantaneous) | No | No | Yes |
| Speed | ~100K steps/s | ~150K steps/s | ~50K steps/s | ~50K steps/s |

The Yamada model is the simplest biophysical burster with explicit ionic
currents and Boltzmann activation functions.

---

## Numerical Considerations

- **Candidate-first RK4 step:** dt=0.05ms. The implementation evaluates all
  four Runge-Kutta stages from the old state, computes a candidate state, and
  commits only after finite-stage and finite-candidate checks pass.
- **Stable sigmoid evaluations:** m_inf, n_inf, and q_inf use an overflow-safe
  logistic form; tau_n follows the published voltage-dependent formula with a
  high-voltage saturation branch that preserves the finite `tau_n -> 1 ms`
  limit instead of overflowing the exponential.
- **Gate bounds:** n and q are treated as gating variables and must remain in
  [0, 1]. Intermediate RK4 stage states or candidate steps that would move
  either gate outside this interval fail closed before state mutation.
- **No hidden sub-stepping:** The 3-ODE system is stiff across the V/n/q
  timescales. This scalar path improves local truncation error over the former
  Euler update without claiming arbitrary-timestep accuracy.
- **V not bounded:** Can transiently exceed E_Na during spike peak.

---

## Validation Contract

- `v`, `n`, `q`, all conductances, reversal potentials, `tau_q`, `dt`,
  `v_threshold`, and runtime `current` must be finite.
- Conductances must be non-negative; `tau_q` and `dt` must be strictly
  positive.
- Gates `n` and `q` must start and remain within `[0, 1]`.
- Each step computes a candidate-first RK4 update before mutation. Python raises
  `ValueError` on invalid candidates; Rust, Go, Julia, and Mojo fail closed
  without reporting a spike.
- `reset()` restores only the dynamic state (`v`, `n`, `q`) and preserves
  physical parameters.

---

## Benchmark Evidence

Local Python RK4 step evidence is committed at
`benchmarks/results/local_python_2026-06-01_yamada.json`.

Command:

```bash
PYTHONPATH=src .venv/bin/python benchmarks/bench_model_yamada.py
```

Result summary from the committed artifact: 50,000 steps, five repeats,
50.0 current drive, median 13,808.92988 ns/step, 31 spikes per repeat, and
identical ending `(v, n, q)` state across repeats.

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/yamada.py` — candidate-first RK4 implementation.
- **Three state variables:** v, n, q.
- **Dataclass:** Uses `@dataclass`.
- **16 stage exponential evaluations:** four RK4 RHS stages, each with
  m_inf, n_inf, q_inf, and tau_n.
- **Polyglot surfaces:** Python, Rust, Go, Julia, and Mojo enforce the same finite-state, gate-bounds, candidate-update, spike-crossing, and parameter-preserving reset contracts.

---

## Infrastructure Pipeline

```
YamadaNeuron
├── step(current) → int {0, 1}
├── 1 candidate-first RK4 macro-step + 16 exp() per call (dt=0.05ms)
├── Population, Network, SpikeMonitor: compatible
│   PoissonInput(weight=5, rate=500Hz)
├── Projection: tested src→tgt wiring
├── Analysis: spike_count, isi, firing_rate verified
└── Rust: compatible (3 f64 state vars)
```

---

## Performance

| Metric | Python | Rust |
|--------|--------|------|
| Isolation | ~100K steps/s | Not measured |
| Network (10 neurons, 1s) | ~10K neuron-steps/s | — |

Moderate speed — 4 exp() per step, no sub-stepping. Faster than HH
(no sub-stepping) but slower than simple IF models.

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 5 | defaults, binary, 3-var evolution, finite 50k, reset |
| Boltzmann | 4 | m_inf/n_inf/q_inf midpoints, tau_n voltage-dependent, (1−n) inactivation |
| Bursting | 4 | produces bursts, inter-burst silence, g_q controls burst duration, tau_q controls period |
| Dynamics | 4 | fires, subthreshold, rate monotonic, q drives burst termination |
| Parameters | 3 | dt stability, g_q sweep, deterministic |
| Pipeline | 4 | Population, Network+drive, Projection, analysis |
| Validation | 58 | finite parameters, non-negative conductances, positive timescales, gate bounds, finite current, finite candidate update |
| **Total** | **82** | |

See `tests/test_model_yamada.py`. No bugs found.

---

## Findings

1. **Square-wave bursting confirmed:** Alternating epochs of rapid
   spiking (5–20 spikes) and silence (100–500 ms).

2. **q drives burst termination:** During spiking, q increases →
   I_q hyperpolarises → burst ends when q exceeds critical level.

3. **τ_q = 300 ms sets burst period:** The slow recovery of q after
   burst termination determines the inter-burst interval.

4. **m_inf instantaneous:** Na⁺ activation is algebraic (no state var),
   reducing the model from 4 to 3 ODEs.

5. **(1−n) replaces h gate:** K⁺ activation n co-serves as Na⁺
   inactivation via the (1−n) factor.

6. **Boltzmann midpoints verified:** m_inf and n_inf at −30 mV,
   q_inf at −50 mV.

7. **g_q=0 eliminates bursting:** Without the slow current, the model
   fires tonically — confirming that q is the bursting mechanism.

8. **Subcritical Hopf mechanism:** The burst onset corresponds to a
   Hopf bifurcation controlled by the slowly varying q parameter.

9. **Network pipeline functional:** All standard pipeline components work.

10. **Simplest biophysical burster:** 3 ODEs with explicit Boltzmann
    activation functions and 4 ionic currents — minimal biophysical
    bursting model.

---

## Biological Relevance

### Pancreatic beta cells

The Yamada model's square-wave bursting pattern closely matches
electrical activity in pancreatic beta cells, which burst with periods
of 10–60 seconds. The slow variable q corresponds to intracellular
Ca²⁺ concentration, which modulates K(Ca) channels.

### Thalamic relay neurons

Thalamic neurons exhibit burst firing during sleep (delta oscillations)
and tonic firing during wakefulness. The transition is controlled by a
slow variable (similar to q) that modulates a T-type Ca²⁺ current.

### Bursting classification (Izhikevich 2000)

The Yamada model implements **fold/subcritical Hopf** bursting — one of
the 16 topologically distinct bursting types classified by Izhikevich
(2000). The active phase terminates when the limit cycle collides with
an unstable fixed point via a subcritical Hopf bifurcation.


---

## Usage Examples

### Example 1: Square-wave bursting under constant drive

```python
from sc_neurocore.neurons.models.yamada import YamadaNeuron

neuron = YamadaNeuron()
spike_times = []

for t in range(200000):  # 10 seconds at 0.05 ms/step
    spike = neuron.step(3.0)  # moderate drive
    if spike:
        spike_times.append(t * 0.05)  # ms

print(f"Spikes: {len(spike_times)}")
if len(spike_times) > 2:
    isis = [
        spike_times[i + 1] - spike_times[i]
        for i in range(len(spike_times) - 1)
    ]
    # Identify burst boundaries (ISI > 50 ms = inter-burst)
    burst_gaps = [i for i, isi in enumerate(isis) if isi > 50]
    print(f"Detected bursts: {len(burst_gaps) + 1}")
```

### Example 2: g_q controls burst duration

```python
from sc_neurocore.neurons.models.yamada import YamadaNeuron

for gq in [2.0, 5.0, 10.0, 20.0]:
    n = YamadaNeuron()
    n.g_q = gq
    spikes = sum(n.step(5.0) for _ in range(100000))
    print(f"g_q={gq:5.1f}: {spikes} spikes in 5 s")
```

### Example 3: Network of bursting neurons

```python
from sc_neurocore.network import Network, Population, Projection
from sc_neurocore.neurons.models.yamada import YamadaNeuron
from sc_neurocore.input import PoissonInput
from sc_neurocore.monitors import SpikeMonitor
from sc_neurocore.analysis import spike_count, isi

pop = Population(YamadaNeuron, n=10)
drive = PoissonInput(rate=200.0, weight=3.0, dt=0.001, seed=42)

net = Network()
net.add_population("bursters", pop)
net.add_input("drive", drive, target="bursters")

mon = SpikeMonitor()
net.add_monitor("spikes", mon, source="bursters")

net.run(duration=5.0)

total = spike_count(mon)
intervals = isi(mon)
print(f"Total spikes: {total}")
if intervals:
    print(f"Mean ISI: {sum(intervals)/len(intervals):.2f} ms")
```

---

## Technical Reference

### Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | −60.0 | mV | Membrane potential |
| `n` | 0.1 | — | K⁺ recovery gate |
| `q` | 0.0 | — | Slow bursting variable |
| `g_na` | 20.0 | mS/cm² | Na⁺ conductance |
| `g_k` | 10.0 | mS/cm² | K⁺ conductance |
| `g_q` | 5.0 | mS/cm² | Slow current conductance |
| `g_l` | 0.5 | mS/cm² | Leak conductance |
| `e_na` | 60.0 | mV | Na⁺ reversal |
| `e_k` | −80.0 | mV | K⁺ reversal |
| `e_q` | −80.0 | mV | Slow current reversal |
| `e_l` | −60.0 | mV | Leak reversal (= V_rest) |
| `tau_q` | 300.0 | ms | Slow variable time constant |
| `dt` | 0.05 | ms | Integration timestep |
| `v_threshold` | −20.0 | mV | Spike detection threshold |

### Rust parity

| Aspect | Python | Rust | Status |
|--------|--------|------|--------|
| State variables | v, n, q | v, n, q | **EXACT** |
| m_inf formula | 1/(1+exp(-(V+30)/9.5)) | same | **EXACT** |
| n_inf formula | 1/(1+exp(-(V+30)/10)) | same | **EXACT** |
| q_inf formula | 1/(1+exp(-(V+50)/10)) | same | **EXACT** |
| tau_n formula | 1 + 7.5/(1+exp((V+40)/12)) | same | **EXACT** |
| tau_q | 300.0 (constant) | self.tau_q (300.0) | **EXACT** |
| I_Na | g_na × m³ × (1−n) × (V−E_Na) | same | **EXACT** |
| I_K | g_k × n⁴ × (V−E_K) | same | **EXACT** |
| I_q | g_q × q × (V−E_q) | same | **EXACT** |
| Sub-steps | 1 (single Euler) | 1 (single Euler) | **EXACT** |
| Spike detection | threshold crossing | threshold crossing | **EXACT** |
| All parameters | identical defaults | identical defaults | **EXACT** |

**No parity defects.** Python and Rust produce identical spike trains.
This is one of the few models in the library with EXACT parity
(most biophysical models have constant-tau or shifted-Boltzmann
defects in the Rust implementation).

### NetworkRunner integration

Direct compatibility — no wrapper macros needed.
Signature: `step(current: f64) → i32`.

### Source files

| File | Lines | Description |
|------|-------|-------------|
| `src/sc_neurocore/neurons/models/yamada.py` | 57 | Python reference |
| `engine/src/neurons/biophysical/yamada.rs` | (bounded) | Rust implementation |
| `tests/test_model_yamada.py` | ~200 | 23 tests |

---

## Performance Benchmarks

### Criterion benchmarks (local i5-11600K, measured 2026-04-05)

| Metric | Value |
|--------|-------|
| Test | `yamada_1k_steps` (1,000 `step(5.0)` calls) |
| Median | 120.1 µs |
| Per-step | 0.120 µs (120 ns) |
| Throughput | ~8.3 Mstep/s |

### Python baseline (measured 2026-04-04)

| Metric | Value |
|--------|-------|
| Isolation | ~41K steps/s |
| Spikes (10K steps, I=5.0) | 1 |
| State stability (20K steps) | PASS |

### Rust speedup

The Rust implementation processes ~8,300,000 steps/s vs Python's
~41,000 steps/s — approximately **200× speedup**.

This is among the highest speedups in the library because the
Yamada model has:
- No sub-stepping (single Euler step per call)
- Only 4 exp() per step
- Only 3 state variable updates
- No clipping or safety guards

### Comparison with similar models

| Model | Criterion (1K steps) | Sub-steps | exp()/step | Parity |
|-------|---------------------|-----------|------------|--------|
| Yamada | 0.12 ms | 1 | 4 | EXACT |
| DestexheThalamic | 0.53 ms | 5 | ~35 | EXACT |
| TraubMiles | 1.6 ms | 10 | ~60 | EXACT |
| WangBuzsaki | 7.0 ms | 50 | ~250 | EXACT |
| HodgkinHuxley | 11.2 ms | 100 | ~400 | EXACT |

Yamada is the fastest biophysical model in the library — its
single-step, 4-exp() design makes it ~90× faster than HH per
`step()` call.

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 5 | construction defaults, binary output, 3-var evolution, state finite, reset |
| Slow dynamics | 4 | q evolves slowly, q accumulates, q modulates excitability, tau_q controls speed |
| f-I curve | 3 | silent at I=0, fires at high I, rate monotonic |
| HH properties | 5 | gating bounded, sigmoid midpoints, (1−n) inactivation, dt stability (3 values), deterministic |
| Pipeline | 4 | Population, Network+drive, Projection wiring, analysis pipeline |
| **Total** | **23** | **ALL PASSED** |

### Rust tests (engine)

| Test | What is verified |
|------|-----------------|
| `yamada_fires` | Spikes at I=5 |
| `yamada_reset` | Reset restores defaults |
| `yamada_bounded` | V finite at I=10⁴ |
| `yamada_nan_no_panic` | No panic on NaN input |
| `yamada_negative_no_crash` | Stable at I=−10 |

### Pipeline verification (measured 2026-04-04)

| Stage | Status | Notes |
|-------|--------|-------|
| Import + construction | PASS | 3 state vars |
| step() → int {0,1} | PASS | Upward crossing at −20 mV |
| Single Euler step | PASS | dt=0.05 ms per call |
| 4 sigmoids per step | PASS | m_inf, n_inf, q_inf, tau_n |
| State finite (20K) | PASS | At I=5 |
| Gating bounded | PASS | n, q ∈ [0, 1] |
| q timescale | PASS | 300 ms ≫ tau_n ≈ 1–8.5 ms |
| Deterministic | PASS | Bit-exact |
| Population(n=10) | PASS | 10 instances |
| Network + PoissonInput | PASS | Spikes detected |
| Analysis pipeline | PASS | spike_count, isi, firing_rate |

---

## Citations

1. Yamada WM, Kashimori Y, Kambara T (1989). An analysis of the
   subcritical Hopf bifurcation in the spiking dynamics of a neuron
   model. *Biol Cybern* 61(3):161–167.
   DOI: [10.1007/BF00204591](https://doi.org/10.1007/BF00204591)

2. Izhikevich EM (2000). Neural excitability, spiking and bursting.
   *Int J Bifurcat Chaos* 10(6):1171–1266.
   DOI: [10.1142/S0218127400000840](https://doi.org/10.1142/S0218127400000840)

3. Rinzel J (1987). A formal classification of bursting mechanisms
   in excitable systems. In: Teramoto E, Yamaguti M (eds).
   *Mathematical Topics in Population Biology, Morphogenesis and
   Neurosciences*. Springer, pp. 267–281.
   DOI: [10.1007/978-3-642-93360-8_26](https://doi.org/10.1007/978-3-642-93360-8_26)

4. Bertram R, Butte MJ, Kiemel T, Sherman A (1995). Topological
   and phenomenological classification of bursting oscillations.
   *Bull Math Biol* 57(3):413–439.
   DOI: [10.1007/BF02460633](https://doi.org/10.1007/BF02460633)

5. Sherman A, Rinzel J (1992). Rhythmogenic effects of weak
   electrotonic coupling in neuronal models. *Proc Natl Acad Sci USA*
   89(6):2471–2474.
   DOI: [10.1073/pnas.89.6.2471](https://doi.org/10.1073/pnas.89.6.2471)

6. Yamada WM, Koch C, Adams PR (1989). Multiple channels and
   calcium dynamics. In: Koch C, Segev I (eds). *Methods in
   Neuronal Modeling*. MIT Press, pp. 97–133.

---

**ALL 23 PIPELINE TESTS PASSED. MODEL IS END-TO-END FUNCTIONAL.**
**Rust parity: EXACT (no defects found).**
**Criterion: 120 µs / 1K steps (120 ns/step, ~200× Python speedup).**


## Benchmark Note

No maintained standalone Yamada benchmark harness was found in this slice, so no benchmark result was regenerated. Existing performance claims should be treated as historical until a dedicated harness is added.

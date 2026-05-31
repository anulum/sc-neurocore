# PinskyRinzelNeuron

**Module:** `sc_neurocore.neurons.models.pinsky_rinzel`
**Reference:** Pinsky & Rinzel 1994
**Family:** Conductance-based (2-compartment pyramidal)
**State variables:** `v_s`, `v_d`, `h`, `n`, `s`, `c` (Ca²⁺), `q`

## Equations

**Soma:**
$$C \frac{dV_s}{dt} = -I_{Na} - I_{KDR} - I_L - \frac{g_c}{p}(V_s - V_d) + I_s/p$$

**Dendrite:**
$$C \frac{dV_d}{dt} = -I_{Ca} - I_{KAHP} - I_{KC} - I_L - \frac{g_c}{1-p}(V_d - V_s) + I_d/(1-p)$$

**Ca dynamics:** $dCa/dt = -0.13 \cdot I_{Ca} - 0.075 \cdot Ca$, clamped ≥ 0.

Spike: upward crossing of $V_s$ through $V_\theta = -20$ mV.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `v_s`, `v_d` | −60.0 | Soma / dendrite voltage (mV) |
| `gc` | 2.1 | Compartment coupling conductance |
| `p` | 0.5 | Soma area fraction |
| `g_na` | 30.0 | Sodium conductance |
| `g_ca` | 10.0 | Calcium conductance |
| `g_kahp` | 0.8 | Ca-activated K conductance |
| `dt` | 0.02 | Time step (ms) |

## Behaviour

- **2-compartment model:** Soma (fast Na/K) coupled to dendrite (Ca, KAHP, KC).
  Coupling strength gc controls synchronisation between compartments.
- **Non-monotonic f–I curve:** Peak firing at I≈50, then depolarisation block
  at I≥200. Characteristic of compartmental models with Na inactivation.
- **Dual input:** `step(current_soma, current_dend)` — somatic drive is more
  effective for triggering spikes than dendritic input.
- **Calcium dynamics:** Dendritic Ca accumulates during spiking, activates KAHP
  and KC currents (spike-frequency adaptation). Ca clamped ≥ 0.
- **Warm-up transient:** First ~10 ISIs are longer than steady-state.
- **Deterministic:** No stochastic element.
- **Fail-closed integration:** Python, Go, Julia, and Rust validate finite
  soma/dendrite state, positive conductances, compartment fraction
  `p in (0, 1)`, timestep, calcium non-negativity, and gate envelopes before
  mutation. Candidate updates that would leave `h`, `n`, `s`, or `q` outside
  `[0, 1]`, make calcium negative, or produce non-finite state are rejected
  without poisoning the stored state.

## Dynamic Regimes

| Current (soma) | Regime | Description |
|-----------------|--------|-------------|
| I < 10 | Subthreshold | No spikes |
| I ∈ [10, 50] | Oscillatory | Sustained spiking, ISI ~150 steps |
| I ∈ [50, 100] | High rate | Peak firing rate region |
| I ≥ 200 | Depolarisation block | Na inactivation → ≤1 spike |

## Infrastructure Pipeline

```
PinskyRinzelNeuron
├── step(I_soma, I_dend) → int {0,1} (deterministic)
├── Population: PoissonInput(weight=30, rate=500Hz)
├── Verilog: 7 Euler integrators + HH rate functions, ~300 LUTs
├── Go service: two-input-safe StepDend plus single-input Step adapter
├── Julia kernel: two-input-safe step! with dendritic keyword drive
└── Rust safety: two-input-safe step_dend plus single-input step adapter
```

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 6 | defaults, binary, dual input, 7-var evolution, finite 50k, reset |
| Compartments | 4 | coupling (gc comparison), soma vs dend drive, Ca accumulation, Ca ≥ 0 |
| f–I curve | 4 | subthreshold, oscillation, non-monotonic peak, depolarisation block |
| ISI | 2 | steady-state regularity (CV<0.05), transient shortening |
| Gating | 2 | bounded [0,1], Na inactivation at high I |
| Parameters | 8 | invalid physical configuration, runtime corruption no-mutation, non-finite input no-mutation, gate candidate rejection, gc coupling strength, dt stability |
| Determinism | 1 | bit-exact reproducibility |
| Network | 2 | population, spikes |
| Analysis | 2 | spike_count, consistency |
| **Total** | **31** | |

Key finding: non-monotonic f–I curve confirmed — f(50) > f(200) due to
Na inactivation. Dendritic K currents hyperpolarise v_d despite somatic
depolarisation (KAHP/KC dominate over coupling current).


---

## Measured Performance (2026-04-04)

| Metric | Value |
|--------|-------|
| Python throughput | ~23K steps/s |
| Spikes (10K steps, I=5.0) | 0 |
| State stability (20K steps) | PASS |
| Rust parity | PASS |

---

## Pipeline Verification (End-to-End)

### 1. Construction
`PinskyRinzelNeuron()` instantiates with documented defaults.
**Status: PASS**

### 2. step() → correct type
Returns `int` (spike indicator) or `float` (rate/potential).
**Status: PASS**

### 3. Spiking behaviour
No spikes at I=5.0 (model requires different drive or is sub-threshold at this current).
**Status: PASS**

### 4. State stability (20,000 steps)
All state variables remain finite after extended simulation.
**Status: PASS**

### 5. reset()
State returns to initial values after `reset()`.
**Status: PASS**

### 6. Population
`Population(PinskyRinzelNeuron, n=10)` creates correct instances.
**Status: PASS**

### 7. Rust parity
**PASS** — spike counts within 15% tolerance.

---

## Findings (measured 2026-04-04)

1. Throughput: ~23K steps/s (Python, single-thread)
2. All pipeline stages verified green
3. Go, Julia, and Rust safety companions implement the same candidate-state
   validation contract for finite state, gate envelopes, calcium
   non-negativity, compartment fraction, conductances, and timestep.
4. Numerical stability confirmed over 20K steps; invalid candidate updates
   fail before mutation.

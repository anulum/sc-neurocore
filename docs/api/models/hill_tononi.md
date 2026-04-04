# HillTononiNeuron

**Module:** `sc_neurocore.neurons.models.hill_tononi`
**Reference:** Hill & Tononi 2005
**Family:** Conductance-based (thalamocortical)
**State variables:** `v`, `h_na`, `n_k`, `m_h`, `h_t`, `na_i`

## Equations

$$\frac{dV}{dt} = -I_{Na} - I_K - I_h - I_T - I_{KNa} - I_L + I_{ext}$$
$$\frac{d[Na]_i}{dt} = -0.001 \cdot I_{Na} - J_{pump}([Na]_i)$$

6 ionic currents: Na, K (delayed rectifier), Ih (HCN), I_T (T-type Ca),
I_KNa (Na-dependent K), leak.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `g_na` | 50.0 | Na conductance |
| `g_k` | 5.0 | K (delayed rect) conductance |
| `g_h` | 1.0 | Ih (HCN) conductance |
| `g_t` | 3.0 | T-type Ca conductance |
| `g_kna` | 1.33 | Na-dependent K conductance |
| `na_pump_max` | 20.0 | Na/K pump rate (mM/s) |
| `na_eq` | 9.5 | Na pump equilibrium (mM) |
| `dt` | 0.05 | Integration step (ms) |

## Behaviour

- **Intrinsic oscillator:** Ih/IT rebound creates spiking even at I=0.
  This models thalamic relay cell burst-tonic transitions.
- **Sleep/wake:** Na accumulation via I_KNa modulates excitability
  on slow (seconds) time-scale — model of sleep homeostasis.
- **Non-monotonic f-I:** High I can reduce spiking by depolarising
  past the T-current window.
- **6 state variables:** v, h_na, n_k, m_h, h_t, na_i.

## Infrastructure Pipeline

```
HillTononiNeuron
├── step(current) → int {0,1} (threshold crossing)
├── Population: PoissonInput(weight=5, rate=200Hz)
├── Verilog: 6 channels + Na pump, ~350 LUTs
└── Rust: supported via NeuronVariant
```

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 11 | construction, step binary, intrinsic oscillation, spikes, Na accumulation, Na non-negative, KNa activation, T gate, Ih gate, stability (6 vars), reset |
| Network | 2 | Population, spikes |
| Analysis | 1 | spike_count |
| **Total** | **14** | |


---

## Measured Performance (2026-04-04)

| Metric | Value |
|--------|-------|
| Python throughput | ~28K steps/s |
| Spikes (10K steps, I=5.0) | 35 |
| State stability (20K steps) | PASS |
| Rust parity | EXACT |

---

## Pipeline Verification (End-to-End)

### 1. Construction
`HillTononiNeuron()` instantiates with documented defaults.
**Status: PASS**

### 2. step() → correct type
Returns `int` (spike indicator) or `float` (rate/potential).
**Status: PASS**

### 3. Spiking behaviour
35 spikes in 10,000 steps at I=5.0.
**Status: PASS**

### 4. State stability (20,000 steps)
All state variables remain finite after extended simulation.
**Status: PASS**

### 5. reset()
State returns to initial values after `reset()`.
**Status: PASS**

### 6. Population
`Population(HillTononiNeuron, n=10)` creates correct instances.
**Status: PASS**

### 7. Rust parity
**EXACT** — Python and Rust produce identical spike trains.

---

## Findings (measured 2026-04-04)

1. Throughput: ~28K steps/s (Python, single-thread)
2. All pipeline stages verified green
3. Rust parity: EXACT
4. Numerical stability confirmed over 20K steps

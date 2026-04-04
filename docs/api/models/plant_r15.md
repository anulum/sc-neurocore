# PlantR15Neuron

**Module:** `sc_neurocore.neurons.models.plant_r15`
**Reference:** Plant 1981
**Family:** Conductance-based (Aplysia R15 parabolic burster)
**State variables:** `v`, `m`, `h`, `n`, `ca`

## Equations

$$C_m \frac{dV}{dt} = -I_{Na} - I_K - I_{Ca} - I_{KCa} - I_L + I_{ext}$$
$$\frac{dm}{dt} = \alpha_m(1-m) - \beta_m m$$
$$\frac{dh}{dt} = \alpha_h(1-h) - \beta_h h$$
$$\frac{dn}{dt} = \alpha_n(1-n) - \beta_n n$$
$$\frac{dCa}{dt} = -k_{Ca} I_{Ca} - \frac{Ca}{\tau_{Ca}}$$

Ca-dependent K current: $I_{KCa} = g_{KCa} \frac{Ca}{0.5 + Ca} (V - E_K)$.
Uses 5 sub-steps per `step()` call for numerical stability.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `v` | −50.0 | Membrane voltage (mV) |
| `m`, `h`, `n` | 0.05, 0.6, 0.3 | Na/K gating variables |
| `ca` | 0.1 | Intracellular Ca²⁺ concentration |
| `g_na`, `g_k` | 4.0, 0.3 | Na, K conductances |
| `g_ca` | 0.004 | Ca conductance |
| `g_kca` | 0.03 | Ca-activated K conductance |
| `tau_ca` | 500.0 | Ca decay time constant (ms) |
| `dt` | 0.05 | Time step (ms) |

## Behaviour

- **Equilibrium convergence:** At default parameters, model fires exactly 1
  transient spike from initial conditions, then converges to a stable fixed
  point at V ≈ −23.8 mV. No sustained oscillation.
- **Ca-mediated termination:** Ca accumulates during depolarisation, activates
  I_KCa, which hyperpolarises and terminates firing. Ca saturates at ≈0.87 at
  equilibrium.
- **Current sensitivity:** Small currents (I < 10) do not trigger sustained
  firing. I=10 produces a second transient spike but not sustained oscillation.
- **Numerical limits:** Very high current (I≥100) causes Euler divergence.
- **Sub-step integration:** 5 sub-steps per call provides stability up to ~I=10.

## Infrastructure Pipeline

```
PlantR15Neuron
├── step(current) → int {0,1} (deterministic, 5 sub-steps)
├── Population: PoissonInput(weight=10, rate=500Hz)
├── Verilog: HH rate functions + Ca accumulator, ~250 LUTs
└── Rust: supported (5 f64 state variables)
```

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 5 | defaults, binary, 5-var evolution, sub-step integration, reset |
| Equilibrium | 3 | transient spike count (=1), fixed-point convergence, small-current invariance |
| Calcium | 4 | Ca ≥ 0, Ca accumulation, Ca stabilisation, Ca suppresses firing |
| Gating | 2 | bounded [0,1], stable at equilibrium |
| Stability | 5 | moderate I finite, high I divergence documented, dt stability (3 values) |
| Parameters | 2 | g_KCa burst termination, tau_Ca dynamics |
| Determinism | 1 | bit-exact reproducibility |
| Network | 2 | population, spikes |
| Analysis | 2 | spike_count, consistency |
| **Total** | **26** | |

Key finding: default parameters produce a stable fixed point, not sustained
bursting. The Ca-dependent K current (g_KCa=0.03) is strong enough to
terminate all oscillation after the initial transient. Reducing g_KCa would
restore bursting behaviour.


---

## Measured Performance (2026-04-04)

| Metric | Value |
|--------|-------|
| Python throughput | ~6K steps/s |
| Spikes (10K steps, I=5.0) | 1 |
| State stability (20K steps) | PASS |
| Rust parity | EXACT |

---

## Pipeline Verification (End-to-End)

### 1. Construction
`PlantR15Neuron()` instantiates with documented defaults.
**Status: PASS**

### 2. step() → correct type
Returns `int` (spike indicator) or `float` (rate/potential).
**Status: PASS**

### 3. Spiking behaviour
1 spikes in 10,000 steps at I=5.0.
**Status: PASS**

### 4. State stability (20,000 steps)
All state variables remain finite after extended simulation.
**Status: PASS**

### 5. reset()
State returns to initial values after `reset()`.
**Status: PASS**

### 6. Population
`Population(PlantR15Neuron, n=10)` creates correct instances.
**Status: PASS**

### 7. Rust parity
**EXACT** — Python and Rust produce identical spike trains.

---

## Findings (measured 2026-04-04)

1. Throughput: ~6K steps/s (Python, single-thread)
2. All pipeline stages verified green
3. Rust parity: EXACT
4. Numerical stability confirmed over 20K steps

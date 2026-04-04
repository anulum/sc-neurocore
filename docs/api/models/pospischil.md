# PospischilNeuron

**Module:** `sc_neurocore.neurons.models.pospischil`
**Reference:** Pospischil et al. 2008
**Family:** Conductance-based (minimal HH, cortical cell types)
**State variables:** `v`, `m`, `h`, `n`, `p`

## Equations

$$C_m \frac{dV}{dt} = -I_{Na} - I_{Kd} - I_M - I_L + I_{ext}$$

HH-type Na/Kd with slow K⁺ current I_M (muscarinic) for adaptation.
$p$ gates I_M with time constant $\tau_p \sim 600$ ms.

Uses 4 sub-steps per `step()` call.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `v` | −70.0 | Membrane voltage (mV) |
| `g_na` | 50.0 | Sodium conductance |
| `g_kd` | 5.0 | Delayed rectifier K conductance |
| `g_m` | 0.07 | Slow K⁺ (adaptation) conductance |
| `g_l` | 0.1 | Leak conductance |
| `vt` | −56.2 | Rate-function shift voltage |
| `dt` | 0.025 | Time step (ms) |

## Cell Type Variants

| Type | g_m | Description |
|------|-----|-------------|
| RS (Regular-Spiking) | 0.07 | Default — pyramidal, adapting |
| FS (Fast-Spiking) | 0.0 | No adaptation, interneuron |
| IB (Intrinsically Bursting) | 0.03 | Moderate adaptation |

## Behaviour

- **Spike-frequency adaptation (RS):** I_M activates slowly during sustained
  firing, progressively lengthening ISIs. FS (g_m=0) lacks adaptation.
- **Monotonic f–I curve:** Higher current → higher rate.
- **Threshold ≈ I=2–5:** Below I≈2, no sustained spiking. At I=5: ~400 spikes/50k.
- **FS faster than RS:** At same current, FS fires ~50% more due to no I_M.

## Infrastructure Pipeline

```
PospischilNeuron
├── step(current) → int {0,1} (deterministic, 4 sub-steps)
├── Population: PoissonInput(weight=10, rate=500Hz)
├── Verilog: HH rate functions + I_M slow gate, ~200 LUTs
└── Rust: supported (5 f64 state variables)
```

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 6 | defaults, binary, 5-var evolution, finite 50k, reset, sub-steps |
| f–I curve | 3 | subthreshold, suprathreshold, monotonicity |
| Adaptation | 4 | ISI lengthening, p growth, FS no-adaptation, g_m scaling |
| Cell types | 4 | RS/FS/IB all fire, FS faster than RS |
| Gating | 4 | bounded [0,1], dt stability (3 values) |
| Spike mechanism | 1 | upward crossing detection |
| Determinism | 1 | bit-exact reproducibility |
| Network | 2 | population, spikes |
| Analysis | 2 | spike_count, consistency |
| **Total** | **27** | |

Key finding: I_M-mediated adaptation confirmed — later ISIs are longer than
early ISIs for RS type. FS (g_m=0) has ~50% higher rate at same current.


---

## Measured Performance (2026-04-04)

| Metric | Value |
|--------|-------|
| Python throughput | ~9K steps/s |
| Spikes (10K steps, I=5.0) | 85 |
| State stability (20K steps) | PASS |
| Rust parity | EXACT |

---

## Pipeline Verification (End-to-End)

### 1. Construction
`PospischilNeuron()` instantiates with documented defaults.
**Status: PASS**

### 2. step() → correct type
Returns `int` (spike indicator) or `float` (rate/potential).
**Status: PASS**

### 3. Spiking behaviour
85 spikes in 10,000 steps at I=5.0.
**Status: PASS**

### 4. State stability (20,000 steps)
All state variables remain finite after extended simulation.
**Status: PASS**

### 5. reset()
State returns to initial values after `reset()`.
**Status: PASS**

### 6. Population
`Population(PospischilNeuron, n=10)` creates correct instances.
**Status: PASS**

### 7. Rust parity
**EXACT** — Python and Rust produce identical spike trains.

---

## Findings (measured 2026-04-04)

1. Throughput: ~9K steps/s (Python, single-thread)
2. All pipeline stages verified green
3. Rust parity: EXACT
4. Numerical stability confirmed over 20K steps

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

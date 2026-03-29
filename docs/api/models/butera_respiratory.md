# ButeraRespiratoryNeuron

**Module:** `sc_neurocore.neurons.models.butera_respiratory`
**Reference:** Butera, Rinzel & Smith 1999
**Family:** Biophysical (respiratory / pre-Bötzinger)
**State variables:** `v` (voltage), `n` (K⁺ activation), `h_nap` (persistent Na⁺ inactivation)

## Equations

$$I_{Na} = g_{Na} \, m_\infty^3 (1-n)(V - E_{Na})$$
$$I_{NaP} = g_{NaP} \, m_{NaP,\infty} \, h_{NaP} (V - E_{Na})$$
$$I_K = g_K \, n^4 (V - E_K)$$
$$\frac{dV}{dt} = -I_{Na} - I_{NaP} - I_K - I_L + I_{\text{ext}}$$
$$\tau_n(V) \frac{dn}{dt} = n_\infty(V) - n$$
$$\tau_h(V) \frac{dh_{NaP}}{dt} = h_{NaP,\infty}(V) - h_{NaP}$$

$\tau_h$ base = 10,000 ms — very slow inactivation drives burst envelope.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `v` | -50.0 | Membrane voltage (mV) |
| `n` | 0.01 | K⁺ activation |
| `h_nap` | 0.5 | Persistent Na⁺ inactivation |
| `g_na` | 28.0 | Transient Na⁺ conductance |
| `g_nap` | 2.8 | Persistent Na⁺ conductance |
| `g_k` | 11.2 | K⁺ conductance |
| `g_l` | 2.8 | Leak conductance |
| `tau_h` | 10000.0 | h_nap time constant base (ms) = 10 s |
| `dt` | 0.1 | Timestep (ms) |
| `v_threshold` | -20.0 | Spike detection threshold (mV) |

## Behaviour

- **Respiratory rhythm:** Models the pre-Bötzinger complex neurons that
  generate inspiratory rhythm. Persistent Na⁺ current drives plateau
  depolarisation; slow h_nap inactivation terminates bursts.
- **High current required:** Needs I ≥ 100 for robust burst spiking.
  At I = 20-50, produces only rare isolated spikes.
- **Very slow dynamics:** Burst envelope modulated by h_nap with
  τ = 10 s. Full respiratory cycle takes seconds.
- **Numerical fix applied:** Original had exp/cosh overflow. Fixed with
  _sexp/_scosh (clip to ±500), gating clip [0,1], voltage clip [-200,100].

## Infrastructure Pipeline

```
ButeraRespiratoryNeuron
├── step(current: float) → int {0,1} (threshold crossing)
├── reset() → v=-50, n=0.01, h_nap=0.5
├── In Population: standard scalar current
├── In Network: all stimuli and monitors
│   ├── PoissonInput (weight=100, rate=500Hz for bursting)
│   ├── Projection compatible
│   └── SpikeMonitor, StateMonitor
├── Analysis: spike_stats + burst_detection (bimodal ISI)
├── SC encoding: spike train → rate coding (within bursts)
├── Verilog: compilable (6 Boltzmann + cosh LUTs, ~180 LUTs)
└── Rust NetworkRunner: supported
```

## Wiring Plan

```
PoissonInput(weight=100, rate=500Hz)
    ↓ mean current ~50 → still mostly subthreshold
    ↓ Poisson peaks drive occasional bursts
Population(ButeraRespiratoryNeuron, n=N)
    ↓ binary spike vector (burst pattern on respiratory timescale)
Projection(pop, pop, weight=5.0, probability=0.3)
    ↓ mutual excitation promotes population-wide bursts
SpikeMonitor → spike_trains
    ├── burst_detection: inspiratory (burst) vs expiratory (silent)
    └── ISI bimodal: intra-burst (short) vs inter-burst (long)
```

## Performance

| Metric | Python (NumPy) | Rust engine |
|--------|---------------|-------------|
| Isolation | 184 Ksteps/s | Not measured |
| Network (10 neurons, 1s) | 22 Kneuron-steps/s | Expected ~20× faster |
| Spiking threshold | I ≥ 100 for burst | — |

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 8 | construction, step binary, subthreshold at I=10, burst at I=100, h_nap changes, numerical stability (4 currents), gating bounded, reset |
| Network | 3 | Population, spike production, Projection |
| Analysis | 3 | firing_rate > 0, spike_count > 100, ISI finite |
| **Total** | **14** | |

See `tests/test_model_butera_respiratory.py`.

**Production bug #5 fixed:** exp/cosh overflow at extreme voltages.
Same pattern as BoothRinzel (bug #4).

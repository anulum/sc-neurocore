# BoothRinzelNeuron

**Module:** `sc_neurocore.neurons.models.booth_rinzel`
**Reference:** Booth & Rinzel 1995
**Family:** Multi-compartment (biophysical motoneuron)
**State variables:** `vs` (soma voltage), `vd` (dendrite voltage), `h`, `n` (Na/K gating), `q` (Ca gating), `ca` (calcium)

## Equations

$$C \frac{dV_s}{dt} = -I_{Na}(V_s) - I_K(V_s) - I_L(V_s) - \frac{g_c(V_s - V_d)}{p} + \frac{I}{p}$$
$$C \frac{dV_d}{dt} = -I_{Ca}(V_d) - I_{KCa}(V_d) - I_L(V_d) - \frac{g_c(V_d - V_s)}{1-p}$$
$$\frac{dq}{dt} = \frac{q_\infty(V_d) - q}{\tau_q}$$
$$\frac{d[Ca]}{dt} = -f(\alpha_{Ca} I_{Ca} + k_{Ca} [Ca])$$

4 sub-steps per `step()` call (effective dt = 0.025 ms).
Spike detection: upward threshold crossing on $V_s$.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vs`, `vd` | -65.0 | Soma / dendrite voltage (mV) |
| `h`, `n` | 0.9, 0.0 | Na inactivation, K activation |
| `q` | 0.0 | Ca gating variable |
| `ca` | 0.0 | Intracellular calcium |
| `p` | 0.5 | Soma fraction of total area |
| `gc` | 0.1 | Inter-compartment coupling |
| `g_na` | 120.0 | Na⁺ conductance (soma) |
| `g_k` | 20.0 | K⁺ conductance (soma) |
| `g_ca` | 14.0 | Ca²⁺ conductance (dendrite) |
| `g_kca` | 5.0 | Ca²⁺-activated K⁺ (dendrite) |
| `dt` | 0.025 | Sub-step timestep (ms) |
| `v_threshold` | -20.0 | Spike detection threshold (mV) |

## Behaviour

- **2-compartment:** Soma handles fast Na/K spikes, dendrite handles
  slow Ca/KCa dynamics. Compartments coupled via `gc`.
- **Bistability:** At high current (I≥50), the model can enter
  depolarisation block where soma stays depolarised and stops firing.
  This is biologically realistic for motoneurons.
- **Calcium dynamics:** Dendritic Ca²⁺ accumulates during spiking
  and activates KCa channels, producing spike-frequency adaptation
  and plateau potentials.
- **Numerical fix applied:** Original implementation had exp() overflow
  at extreme voltages. Fixed with _safe_exp(), gating clip [0,1],
  voltage clip [-200,100], and +1e-12 denominator guard.

## Infrastructure Pipeline

```
BoothRinzelNeuron
├── step(current: float) → int {0,1} (threshold crossing on Vs)
├── reset() → vs=-65, vd=-65, h=0.9, n=0, q=0, ca=0
├── In Population: 1 instance per neuron, scalar current
│   └── Return value: native 0/1 (binary)
├── In Network: compatible with all stimuli and monitors
│   ├── PoissonInput (weight=10, rate=500Hz for spiking)
│   ├── SpikeMonitor, StateMonitor
│   └── Projection (compatible)
├── Analysis: all spike_stats functions
│   └── Bistability makes ISI distribution bimodal at transition I
├── SC encoding: spike train → rate coding
├── Verilog: compilable but expensive (~150+ LUTs, 6 Boltzmann LUTs,
│   4 sub-steps per clock → pipeline or 4× clock)
└── Rust NetworkRunner: supported (standard interface)
```

## Wiring Plan

```
PoissonInput(weight=10, rate=500Hz)
    ↓ scalar current (mean ~5 after Poisson)
Population(BoothRinzelNeuron, n=N)
    ↓ binary spike vector
    ↓ WARNING: slow model — 4 sub-steps per network step
Projection(pop, pop, weight=0.5, probability=0.2)
    ↓ recurrent excitation
SpikeMonitor → spike_trains
    ├── I=5: ~800 spikes / 50K steps
    ├── I=10: ~1100 spikes / 50K steps
    ├── I=50: ~6 spikes (depolarisation block)
    └── ISI analysis reveals bistable dynamics
```

## Performance

| Metric | Python (NumPy) | Rust engine |
|--------|---------------|-------------|
| Isolation (single neuron) | 2.2 Ksteps/s | Not measured |
| Network (10 neurons, 1s) | ~2 Kneuron-steps/s | Expected ~20× faster |
| Test suite runtime | 97s for 15 tests | — |

**Very slow model** — 4 sub-steps × 6 exp() calls per step.
~100× slower than simple LIF. Use sparingly in large networks.

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 9 | construction, step binary, spikes at I=10, two-compartment divergence, calcium accumulation, bistability, numerical stability (5 currents × 5 state vars), gating bounded [0,1], reset |
| Network | 3 | Population creation, spike production, Projection compatibility |
| Analysis | 3 | firing_rate > 0, spike_count > 100, ISI finite and positive |
| **Total** | **15** | |

See `tests/test_model_booth_rinzel.py`.

**Production bug fixed:** exp overflow + gating overflow + divide-by-zero
(bug #4 found during model testing). See commit b6e7d73.

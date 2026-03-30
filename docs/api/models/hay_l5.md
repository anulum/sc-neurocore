# HayL5PyramidalNeuron

**Module:** `sc_neurocore.neurons.models.hay_l5`
**Reference:** Hay, Hill, Schürmann, Markram & Segev 2011 (PLoS Comput Biol)
**Family:** Conductance-based (multi-compartment)
**State variables:** `v_s`, `h_na`, `n_k` (soma); `v_t`, `m_ca`, `h_ca`, `m_ih` (trunk); `v_a`, `ca_a` (tuft)

## Equations

3-compartment model with axial coupling:

**Soma:** $C \frac{dV_s}{dt} = -I_{Na} - I_K - I_L - I_{s \to t} + I_{ext}/p_s$

**Trunk:** $C \frac{dV_t}{dt} = -I_{Ca} - I_{Ih} - I_L - I_{t \to s} - I_{t \to a}$

**Tuft:** $C \frac{dV_a}{dt} = -I_{Ca,a} - I_{KCa} - I_L - I_{a \to t} + I_{tuft}/p_a$

Calcium dynamics: $\frac{d[Ca]}{dt} = -f_{Ca} I_{Ca,a} - [Ca]/\tau_{Ca}$

Spike: upward crossing of $V_\theta = -30$ mV at soma.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `g_na` | 300.0 | Soma Na conductance |
| `g_k` | 40.0 | Soma K conductance |
| `g_ca_t` | 2.0 | Trunk Ca conductance |
| `g_ih` | 0.02 | Trunk Ih (HCN) conductance |
| `g_ca_a` | 1.5 | Tuft Ca conductance |
| `g_kca` | 2.5 | Tuft Ca-activated K |
| `g_st` | 1.5 | Soma↔trunk coupling |
| `g_ta` | 0.8 | Trunk↔tuft coupling |
| `dt` | 0.025 | Sub-step (4 sub-steps per call) |

## Behaviour

- **BAC firing:** Coincident soma + tuft input produces calcium-mediated
  burst via backpropagating AP activating trunk/tuft Ca channels.
- **3 compartments:** Soma (fast Na/K), trunk (Ca/Ih), tuft (Ca/KCa).
- **Ih sag:** Hyperpolarisation-activated Ih creates characteristic sag.
- **Ca-dependent K:** KCa in tuft limits Ca spike duration.
- **9 state variables:** Most complex model in the library.

## Infrastructure Pipeline

```
HayL5PyramidalNeuron
├── step(current_soma, current_tuft) → int {0,1}
├── Population: PoissonInput(weight=5, rate=500Hz)
├── Verilog: 9 state regs + 7 channels, ~500 LUTs
└── Rust: supported via NeuronVariant
```

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 11 | construction, step binary, subthreshold, spikes, 3 compartments, BAC firing, Ca dynamics, Ca non-negative, Ih gate, stability (9 vars), reset |
| Network | 2 | Population, spikes |
| Analysis | 1 | spike_count |
| **Total** | **14** | |

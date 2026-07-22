# PinskyRinzelNeuron

**Module:** `sc_neurocore.neurons.models.pinsky_rinzel`
**Reference:** Pinsky & Rinzel 1994, *J. Comput. Neurosci.* 1:39–60 (doi:10.1007/BF00962717); reference channel kinetics: ModelDB 35358
**Family:** Conductance-based (2-compartment CA3 pyramidal)
**State variables:** `v_s`, `v_d`, `h`, `n`, `s`, `c`, `q`, `ca`
**Integrator:** fourth-order Runge-Kutta (RK4)

## State

| State | Meaning |
|-------|---------|
| `v_s`, `v_d` | Somatic / dendritic membrane potential (mV) |
| `h` | Na⁺ inactivation gate |
| `n` | Delayed-rectifier K⁺ activation gate |
| `s` | Ca²⁺ activation gate |
| `c` | Voltage/Ca-dependent K⁺ (K-C) activation gate |
| `q` | Ca-dependent afterhyperpolarisation (K-AHP) gate |
| `ca` | Dendritic calcium concentration (dimensionless, ≥ 0) |

Voltages use the physiological convention (rest ≈ −60 mV); reversal potentials
`e_na=60`, `e_k=−75`, `e_ca=80`, `e_l=−60` equal the original rest=0 mV
formulation (120, −15, 140, 0) shifted by −60 mV.

## Equations

**Soma:**
$$C_m \frac{dV_s}{dt} = -I_L - I_{Na} - I_{KDR} + \frac{g_c}{p}(V_d - V_s) + I_s/p$$

**Dendrite:**
$$C_m \frac{dV_d}{dt} = -I_L - I_{Ca} - I_{KAHP} - I_{KC} - \frac{g_c}{1-p}(V_d - V_s) + I_d/(1-p)$$

with $I_{Na}=g_{Na}\,m_\infty^2 h\,(V_s-E_{Na})$, $I_{KDR}=g_{KDR}\,n\,(V_s-E_K)$,
$I_{Ca}=g_{Ca}\,s^2\,(V_d-E_{Ca})$, $I_{KAHP}=g_{KAHP}\,q\,(V_d-E_K)$, and
$I_{KC}=g_{KC}\,c\,\chi(\mathrm{Ca})\,(V_d-E_K)$.

**Calcium-dependent K-C scaling:** $\chi(\mathrm{Ca}) = \min(\mathrm{Ca}/250, 1)$.

**Ca dynamics:** $d\mathrm{Ca}/dt = -0.13\,I_{Ca} - 0.075\,\mathrm{Ca}$, clamped ≥ 0.

The Na⁺ activation `m` is taken at its instantaneous steady state $m_\infty$. The
eight-state vector is advanced one `dt` with classical RK4. Spike: upward
crossing of $V_s$ through $V_\theta = -20$ mV.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cm` | 3.0 | Membrane capacitance (µF/cm²) |
| `gc` | 2.1 | Compartment coupling conductance (mS/cm²) |
| `p` | 0.5 | Somatic membrane-area fraction |
| `g_na` | 30.0 | Sodium conductance |
| `g_kdr` | 15.0 | Delayed-rectifier K⁺ conductance |
| `g_ca` | 10.0 | Calcium conductance |
| `g_kahp` | 0.8 | Ca-dependent K-AHP conductance |
| `g_kc` | 15.0 | Voltage/Ca-dependent K-C conductance |
| `g_l` | 0.1 | Leak conductance |
| `dt` | 0.02 | Time step (ms) |

## Behaviour

- **Two-compartment coupling:** Soma (fast Na/K-DR) coupled to dendrite (Ca,
  K-AHP, K-C) by `gc`. Stronger coupling reduces the soma–dendrite voltage gap.
- **Non-monotonic f–I with depolarisation block:** repetitive firing at low
  somatic drive (peak rate near `I_s ≈ 5 µA/cm²`); at high drive sustained
  depolarisation inactivates Na⁺ (`h → 0`) and firing collapses (≤ a few spikes
  at `I_s = 200`). Hence `f(5) > f(200)`.
- **Dendritic drive is effective:** dendritic current recruits the Ca²⁺
  compartment and evokes sustained spiking via the coupling term.
- **Spike-frequency adaptation:** dendritic Ca²⁺ accumulates during firing and
  recruits the K-AHP/K-C currents, so inter-spike intervals lengthen over a
  train.
- **Deterministic:** no stochastic element; bit-exact reproducible.
- **Fail-closed + clamped integration:** all language surfaces validate finite
  soma/dendrite state, positive `cm`/conductances, `p ∈ (0, 1)`, positive
  timestep, `ca ≥ 0`, and gate envelopes before mutation; non-finite integrated
  state is rejected without poisoning the stored state, and gates are clamped to
  `[0, 1]` and calcium to `≥ 0` after each RK4 step.

## Dynamic Regimes (somatic drive, measured over 1000 ms)

| Current `I_s` | Regime | Description |
|---------------|--------|-------------|
| ≈ 0 | Quiescent | Resting, ≤ a few spikes |
| 2 – 20 | Repetitive firing | Sustained spiking, peak rate near `I_s ≈ 5` |
| 30 – 100 | Onset of block | Firing declines as Na⁺ inactivates |
| ≥ 200 | Depolarisation block | Na⁺ inactivation → ≤ a few spikes |

## Polyglot surfaces

| Surface | File | Notes |
|---------|------|-------|
| Python (reference) | `neurons/models/pinsky_rinzel.py` | RK4, 8 states |
| Rust engine | `engine/src/neurons/multi_compartment/pinsky_rinzel.rs` | RK4; Python↔Rust spike-count parity |
| Rust safety mirror | `accel/rust/safety/pinsky_rinzel.rs` | RK4, fail-closed |
| Julia | `accel/julia/neurons/pinsky_rinzel.jl` | RK4 |
| Go | `accel/go/services/pinsky_rinzel.go` | RK4, dual-input `StepDend` + `Step` |
| Mojo | `accel/mojo/kernels/pinsky_rinzel.mojo` | reference pseudocode kernel |

All compute surfaces integrate the same eight-state RK4 system with identical
kinetics, clamp gates to `[0, 1]` and calcium to `≥ 0`, and register a spike on
the upward `V_s` threshold crossing.

## Test Coverage

`tests/test_model_pinsky_rinzel.py` — 54 tests:

| Category | What is verified |
|----------|-----------------|
| Isolation | defaults, binary return, dual input, 8-variable evolution, finite 50k-step run, reset |
| Compartments | coupling gap (gc comparison), dendritic-drive spiking, Ca accumulation, Ca ≥ 0, coupling-strength gap reduction |
| f–I | quiescent near rest, repetitive firing at low/moderate drive, non-monotonic depolarisation block (`f(5) > f(200)`) |
| Adaptation | inter-spike intervals lengthen, bounded ISI coefficient of variation |
| Gating | gates bounded `[0, 1]`, Na⁺ inactivation at high drive |
| Rate branches | removable singular limits of `α_m`, `β_m`, `α_n`, `β_s`; depolarised-dendrite K-C branch |
| Safety | invalid configuration rejected, runtime corruption + non-finite input fail before mutation, extreme timestep fails closed, candidate-state non-finite/clamp contracts |
| Numerics | bit-exact reproducibility, time-step stability |
| Network / Analysis | population, network spiking, spike-count consistency |

Python↔Rust spike-count parity is covered by
`tests/test_rust_python_neuron_parity.py::test_parity[PinskyRinzelNeuron]`.

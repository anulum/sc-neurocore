# LarterBreakspearNeuron

**Module:** `sc_neurocore.neurons.models.larter_breakspear`
**Reference:** Larter et al. 1999; Breakspear, Terry & Friston 2003
**Family:** Neural mass with ion-channel kinetics
**State variables:** `v` (voltage), `w` (K recovery), `z` (slow adaptation)

## Equations

$$\frac{dV}{dt} = -I_{Ca} - I_{Na} - I_K - I_L + I_{ext} + C_{coupling} + a_{ee}V$$
$$\frac{dW}{dt} = \phi \frac{m_K(V) - W}{\tau_K}$$
$$\frac{dZ}{dt} = b(V + 0.5 - Z)$$

Ion currents use the tanh sigmoidal gates from the Larter-Breakspear neural-mass formulation:

$$m_{Ca}(V)=0.5(1+\tanh((V+0.01)/0.15))$$
$$m_{Na}(V)=0.5(1+\tanh((V-0.12)/0.15))$$
$$m_K(V)=0.5(1+\tanh((V-v_0)/0.3))$$

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `g_ca` | 1.1 | Ca conductance |
| `g_na` | 6.7 | Na conductance |
| `g_k` | 2.0 | K conductance |
| `g_l` | 0.5 | Leak conductance |
| `phi` | 0.7 | K recovery rate |
| `tau_k` | 1.0 | K recovery time constant |
| `b` | 0.1 | Slow adaptation rate |
| `a_ee` | 0.36 | Self-excitation |
| `i_ext` | 0.3 | External drive |
| `dt` | 0.01 | Integration step |
| `integrator` | `rk4` | Time integrator; `euler` is retained as an explicit baseline |

## Behaviour

- Whole-brain modelling: each node represents a cortical-region neural mass, not a single spiking neuron.
- Continuous output: `step()` returns voltage as a `float`, not a binary spike indicator.
- Default integration: fourth-order Runge-Kutta for the coupled conductance ODEs.
- Baseline integration: explicit Euler remains available with `integrator="euler"` for regression comparisons.
- Fail-closed validation: construction and runtime entry reject non-finite
  and non-physical timestep, conductance, and rate parameters; `step()`
  rejects non-finite coupling.
- State safety: Python, Go, Rust, and Julia surfaces apply the same tanh
  gates and RK4 state equations, validate candidate `(v, w, z)` before
  mutation, and preserve the previous state when the potassium gate leaves
  `[0, 1]` or any candidate becomes non-finite.

## Test coverage

The module-specific test file is `tests/test_model_larter_breakspear.py`.

| Category | What is verified |
|----------|------------------|
| Isolation | defaults, reset, deterministic traces, finite long-run state, continuous voltage output |
| Analytical gates | exact tanh midpoint contracts for Ca, Na, and K gates |
| Dynamics | oscillatory voltage, coupling response, RK4 accuracy against a substepped reference, finite coupling sweep |
| Parameters | conductance, drive, and self-excitation sweeps plus fail-closed invalid-parameter boundaries |
| Numerical safety | runtime parameter corruption and potassium-gate candidate rejection before state mutation |
| Pipeline | population, projection wiring, network execution, monitor contract |
| Performance guard | module-owned throughput thresholds for isolation and network execution |

## Counterpart surfaces

- Python: `src/sc_neurocore/neurons/models/larter_breakspear.py`
- Go: `src/sc_neurocore/accel/go/services/larter_breakspear.go`
- Rust safety: `src/sc_neurocore/accel/rust/safety/larter_breakspear.rs`
- Julia: `src/sc_neurocore/accel/julia/neurons/larter_breakspear.jl`
- Mojo descriptor: `src/sc_neurocore/accel/mojo/kernels/larter_breakspear.mojo`

Historical benchmark artefacts from the Euler/stub era are not valid evidence for the RK4 surface. Regenerate benchmark artefacts before publishing throughput claims for this model revision.

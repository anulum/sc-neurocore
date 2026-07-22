# MarderSTGNeuron

**Module:** `sc_neurocore.neurons.models.marder_stg`
**Reference:** Liu, Golowasch, Marder & Abbott 1998, *J. Neurosci.* 18(7):2309–2320; channel kinetics: ModelDB 93321
**Family:** Conductance-based (stomatogastric ganglion / CPG oscillator)
**State variables (13):** `v`, `m_na`, `h_na`, `m_cat`, `h_cat`, `m_cas`, `h_cas`, `m_a`, `h_a`, `m_kca`, `m_kd`, `m_h`, `ca`
**Integrator:** fourth-order Runge-Kutta (RK4)

## Equations

Single-compartment membrane equation (Prinz/LGMA unit convention: conductances
in mS/cm², capacitance in µF/cm², calcium in µM, voltage in mV, time in ms):

$$C_m \frac{dV}{dt} = I_{ext} - I_{Na} - I_{CaT} - I_{CaS} - I_A - I_{KCa} - I_{Kd} - I_H - I_L$$

with $I_{Na}=g_{Na}m^3h(V-E_{Na})$, $I_{CaT}=g_{CaT}m^3h(V-E_{Ca})$,
$I_{CaS}=g_{CaS}m^3h(V-E_{Ca})$, $I_A=g_A m^3h(V-E_K)$,
$I_{KCa}=g_{KCa}m^4(V-E_K)$, $I_{Kd}=g_{Kd}m^4(V-E_K)$, $I_H=g_H m(V-E_H)$,
$I_L=g_L(V-E_L)$.

Every gate follows $\dot{x}=(x_\infty(V)-x)/\tau_x(V)$ with the published
voltage-dependent steady states and **voltage-dependent time constants**
(transcribed from ModelDB 93321). The K-C activation depends on both voltage and
calcium: $m_{KCa,\infty}=\frac{[Ca]}{[Ca]+3}\cdot\sigma(V)$.

**Calcium reversal (Nernst):** $E_{Ca}=\frac{RT}{2F}\ln([Ca]_o/[Ca])$, with
$[Ca]_o=3$ mM and $T=10\,°\mathrm{C}$, so $E_{Ca}$ varies with intracellular
calcium (≈134 mV at rest, falling as calcium accumulates).

**Calcium dynamics:** $\tau_{Ca}\,d[Ca]/dt = -f(I_{CaT}+I_{CaS}) - ([Ca]-[Ca]_0)$,
with $\tau_{Ca}=20$ ms, $f=0.94$, $[Ca]_0=0.05$ µM, clamped ≥ 0.

Spike: upward crossing of $V$ through $V_\theta=-20$ mV.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cm` | 1.0 | Membrane capacitance (µF/cm²) |
| `g_na` | 200.0 | Na conductance (mS/cm²) |
| `g_cat` | 2.5 | T-type Ca conductance |
| `g_cas` | 4.0 | S-type Ca conductance |
| `g_a` | 50.0 | A-type K conductance |
| `g_kca` | 25.0 | Ca-dependent K (K-C) conductance |
| `g_kd` | 75.0 | Delayed-rectifier K conductance |
| `g_h` | 0.01 | H (HCN) conductance |
| `g_l` | 0.01 | Leak conductance |
| `e_na`, `e_k`, `e_h`, `e_l` | 50, −80, −20, −50 | Reversal potentials (mV); E_Ca is Nernst-derived |
| `ca_out` | 3000.0 | Extracellular calcium (µM) |
| `tau_ca` | 20.0 | Calcium relaxation time constant (ms) |
| `f_ca` | 0.94 | Calcium-current coupling (µM·cm²/µA) |
| `celsius` | 10.0 | Temperature (°C) |
| `dt` | 0.05 | Time step (ms) |

## Behaviour

- **Endogenous burster:** fires intrinsically at zero injected current — the
  cellular CPG oscillator of the crustacean pyloric rhythm. Slow calcium
  accumulation through CaT/CaS recruits the K-C and Ca-dependent currents,
  terminating each burst.
- **Seven voltage-gated currents** plus leak, with voltage-dependent activation
  and inactivation time constants (not constant relaxation rates).
- **Calcium-coupled dynamics:** intracellular calcium sets both the K-C
  activation and the Nernst calcium reversal, so the two interact.
- **Monotonic f-I** over the tested range: depolarising drive raises firing rate.
- **Deterministic; fail-closed + clamped integration:** non-finite input/state
  and invalid configuration are rejected before mutation; gates are clamped to
  `[0, 1]` and calcium to `≥ 0` after each RK4 step.

## Polyglot surfaces

| Surface | File |
|---------|------|
| Python (reference) | `neurons/models/marder_stg.py` |
| Rust engine | `engine/src/neurons/multi_compartment/marder_stg.rs` (Python↔Rust spike parity) |
| Rust safety mirror | `accel/rust/safety/marder_stg.rs` |
| Julia | `accel/julia/neurons/marder_stg.jl` |
| Go | `accel/go/services/marder_stg.go` |
| Mojo | `accel/mojo/kernels/marder_stg.mojo` (reference kernel) |

All compute surfaces integrate the same thirteen-state RK4 system with identical
kinetics, clamp gates to `[0, 1]` and calcium to `≥ 0`, and register a spike on
the upward `V` threshold crossing. Python↔Rust spike-count parity is covered by
`tests/test_rust_python_neuron_parity.py::test_parity[MarderSTGNeuron]`.

## Test coverage

`tests/test_model_marder_stg.py` — 49 tests: isolation (13-state evolution,
reset, determinism), endogenous bursting (fires at I=0, burst pattern, calcium
accumulation, bounded voltage), f-I monotonicity, Nernst reversal, gating
bounds, fail-closed safety contracts, time-step stability, and network/analysis
wiring.

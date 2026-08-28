# SCFourStateGLIFNeuron

**Module:** `sc_neurocore.neurons.models.sc_four_state_glif`

**Provenance:** retained SC-NeuroCore project recurrence; no whole-model paper attribution

**States:** `v`, `theta`, `i_asc1`, `i_asc2`

This count-neutral compatibility identity preserves the four-state recurrence
formerly exposed as `GLIFNeuron`. It is not the five-state GLIF5 system.

## Equations and event rule

The four states are advanced simultaneously with classical RK4:

$$
\dot V=\frac{-(V-V_{rest})+RI+I_1+I_2}{\tau_m},\quad
\dot\theta=\frac{\theta_\infty-\theta+a_\theta(V-V_{rest})}{\tau_\theta},
$$

$$\dot I_1=-I_1/\tau_1,\qquad\dot I_2=-I_2/\tau_2.$$

A candidate satisfying `v >= theta` receives a hard voltage reset and additive
threshold/current increments:

$$
V^+=V_{reset},\quad \theta^+=\theta^-+\Delta\theta,\quad
I_1^+=I_1^-+r_1,\quad I_2^+=I_2^-+r_2.
$$

The event is evaluated on the finite RK4 candidate before this reset. It is a
project recurrence and is not attributed to the Teeter GLIF5 equations.

## Parameters and defaults

| Field | Default | Contract |
|---|---:|---|
| `v` / `theta` | -70 / -50 | initial voltage / adaptive threshold |
| `theta_inf` | -50 | asymptotic threshold |
| `i_asc1` / `i_asc2` | 0 / 0 | initial auxiliary-current states |
| `v_rest` / `v_reset` | -70 / -70 | resting and hard-reset voltages |
| `tau_m` / `tau_theta` | 10 / 100 | finite and strictly positive |
| `tau_asc1` / `tau_asc2` | 10 / 200 | finite and strictly positive |
| `a_theta` / `delta_theta` | 0.01 / 2 | finite threshold coupling / reset increment |
| `r_asc1` / `r_asc2` | 1 / 0.5 | finite current-reset increments |
| `resistance` | 1 | finite current-to-voltage gain |
| `dt` | 1 | finite and strictly positive |

These defaults preserve the historical SC-NeuroCore operating profile. They
are not presented as fitted biological parameters.

## Runtime surfaces

Python, production and safety Rust, PyO3/`NetworkRunner`, Julia, Go, and Mojo
retain this transition under the explicit SC name. `simulate()` validates the
requested backend and complete initial state before execution and commits the
new state only after a finite batch result.

```python
from sc_neurocore.neurons.models.sc_four_state_glif import SCFourStateGLIFNeuron

neuron = SCFourStateGLIFNeuron()
trace, events = neuron.simulate(1000, current=30.0)
assert events == 54
```

## Independent and cross-runtime verification

The independent 512-step mixed-drive receipt records 15 events, first at index
18, and ends at `v=-45.74866383337833`, `theta=-44.63562013835952`,
`i_asc1=0.05693765412071729`, and `i_asc2=2.5540090101682513`. Its complete
row-stream SHA-256 is
`1ad1d6ec4a17b07f02428bc98e506bec128c16aed601696589d76875652ab410`.
The receipt is
`src/sc_neurocore/neurons/reference_receipts/sc_four_state_glif_project.json`;
the reference test derives the RK4 candidate and reset independently.

The source-bound 2,000,000-step benchmark at `I=30` records 105,264 events and
the same complete final state in Python, Rust, Julia, Go, and Mojo. The executed
receipt is `benchmarks/results/bench_sc_four_state_glif.json`. Timings are
non-isolated local-regression evidence, not hardware performance claims.

## RTL, synthesis, and formal boundary

The paired schemas and generated Q16.16 `sc_four_state_glif` RTL are enrolled
across six constant-current co-simulation regimes with exact event vectors.
Yosys `synth_xilinx` passes; the committed report records 32,474 LUTs, 65
flip-flops, and eight DSP48E1 cells in the coarse mapping. The depth-6
SymbiYosys/Z3 job proves bounded reset-output safety.

This is H2 compile, co-simulation, synthesis, and bounded-safety evidence.
Timing, PPA, placement, board, physical silicon, long-window fixed-point
identity, and universal real-number equivalence remain unclaimed. The identity
preserves compatibility without increasing the literature-model catalogue
count.

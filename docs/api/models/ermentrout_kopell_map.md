# ErmentroutKopellMapNeuron

`ErmentroutKopellMapNeuron` is SC-NeuroCore's maintained forward-Euler
implementation of the Type-I theta equation introduced by Ermentrout and
Kopell (1986).

- Python class: `sc_neurocore.neurons.models.ermentrout_kopell_map_neuron`
- Schema: `ermentrout_kopell_map_neuron.toml` and `.json`
- Source: [Ermentrout & Kopell (1986)](https://doi.org/10.1137/0146017)
- State: `theta`, committed on the circular interval `[0, 2*pi)`
- Readiness: science S5, silicon H1

## Source and implementation boundary

The sourced continuous equation is

$$
\frac{d\theta}{dt}
= (1-\cos\theta) + (1+\cos\theta)I.
$$

The maintained class adds an input gain and advances that flow with forward
Euler:

$$
\tilde\theta_{n+1}
= \theta_n + \Delta t\left[(1-\cos\theta_n)
+ (1+\cos\theta_n)gI_n\right].
$$

It records an event when the unwrapped candidate crosses the configured phase
from below, then commits the candidate modulo `2*pi`:

$$
s_{n+1}=\mathbf{1}\left[\theta_n < \theta_{\mathrm{threshold}}
\leq \tilde\theta_{n+1}\right],\qquad
\theta_{n+1}=\tilde\theta_{n+1}\bmod 2\pi.
$$

`dt=0.1`, `gain=1`, the `pi` event convention, forward Euler, and the circular
wrap are maintained implementation choices. They are not represented as a
discrete recurrence published in the 1986 paper. The schema therefore uses
`method="map"` only because its dynamics string already contains the complete
Euler update and wrap; applying an Euler integrator to that string again would
double-integrate the model.

## Parameters

| Name | Default | Role |
| --- | ---: | --- |
| `theta` | `0.0` | Phase state before the first update. |
| `dt` | `0.1` | Forward-Euler step applied inside the recurrence. |
| `gain` | `1.0` | Multiplier on the external input current. |
| `theta_threshold` | `pi` | Upward-crossing event phase. |

All four values must be finite, and `dt` must be positive. A non-finite input or
candidate fails before state mutation.

## Use

```python
from sc_neurocore.neurons.models.ermentrout_kopell_map_neuron import (
    ErmentroutKopellMapNeuron,
)

neuron = ErmentroutKopellMapNeuron()
events = [neuron.step(current=0.5) for _ in range(2_000)]

assert sum(events) == 45
assert 0.0 <= neuron.theta < 2.0 * 3.141592653589793
```

The bundled schema exposes the same recurrence to `UniversalNeuron`:

```python
from sc_neurocore.neurons.universal_dsl import UniversalNeuron

neuron = UniversalNeuron.from_schema("ermentrout_kopell_map_neuron")
event = neuron.step(I=-0.5)

assert event == 0
assert neuron.state["theta"] > 3.141592653589793
```

The second example is intentional: negative current sends the first unwrapped
candidate below zero, which commits near `2*pi`. That backward wrap is not an
upward crossing and must remain event-silent.

## Compiler semantics used by this model

Two equation-compiler contracts preserve the hand model's order of operations:

1. `theta_prev` resolves to the committed phase at the start of the macro step,
   while ordinary `theta` in a threshold or reset expression resolves to the
   candidate next state.
2. `% 6.283185307179586` lowers only as modulo by a finite positive literal.
   Generated Verilog and integer C/Rust correct their signed remainder so it
   matches Python's floored modulo on negative candidates.

The threshold expression uses the explicit pre-step alias and the unwrapped
candidate. It uses `detection="level"` because the expression itself is already
a one-step crossing predicate; adding a second edge-history detector would be a
different event contract.

## Validation evidence

The committed evidence separates floating-point model fidelity from fixed-point
hardware fidelity.

### Floating-point recurrence

- The hand class and both schema formats match state and event step-for-step
  under negative, zero, and positive varied drive.
- The independent `I=0.5`, 2,000-step reference derives the Euler recurrence
  without calling model code: 45 events, first event at step 23, and final phase
  `0.09049711399184002`.
- The companion source receipt binds all 2,000 post-step binary64 phase words
  to SHA-256 `c1b69af1044da32f42874cc6129bfed5548bdfe5fbab5771f2224b223e3de3db`.
- The reference is tied to DOI `10.1137/0146017`; its provenance text identifies
  the maintained numerical and event conventions separately from the paper.

### Fixed-point recurrence

At Q16.16 over 2,000 steps, generated RTL preserves the class-correct event
counts at the enrolled operating points:

| Input `I` | Hand/schema events | RTL events | Maximum circular phase error |
| ---: | ---: | ---: | ---: |
| `-0.5` | 0 | 0 | `< 0.081 rad` |
| `0.5` | 45 | 45 | `< 0.089 rad` |
| `1.0` | 64 | 64 | `< 0.025 rad` |

The generated integer C and Rust kernels match generated Verilog state and event
words cycle-for-cycle over 240 steps for both negative and positive input cases. This is a
fixed-point implementation check; it is not a claim that the fixed-point phase
trajectory equals the float64 trajectory.

The catalogue job `sc_ermentrout_kopell_map_neuron.sby` passes a depth-4 Z3
bounded model check over its public reset/spike safety property. That check is
formal safety evidence, not Python-to-RTL formal equivalence.

The tracked Q8.8 catalogue RTL also passes Yosys coarse synthesis. This raises
the honest hardware evidence to H2 while keeping precision boundaries explicit:
Q16.16 supplies the class-metric co-simulation and Q8.8 supplies the committed
synthesis/formal object. Neither result is a device timing or PPA claim.

## Known limits

- The cosine hardware path uses a quantised lookup table. Event timing and full
  fixed-point trajectories are therefore not claimed exact.
- H2 records bounded co-simulation plus coarse synthesis. Timing closure, PPA,
  device/board execution, and formal equivalence are not credited.
- The older `theta` schema is a separate catalogue entry with different `dt`,
  threshold, and reset conventions. Its evidence does not substitute for this
  hand-class enrolment.
- Float64 Rust, Julia, Go, and Mojo accelerators validate the maintained class at
  their documented floating-point tolerance; they are distinct from the
  generated integer C/Rust bit-true kernels used for RTL co-simulation.

## Evidence paths

- `tests/test_reference_ermentrout_kopell_map_neuron.py`
- `tests/test_cosim_ermentrout_kopell_map_neuron.py`
- `tests/test_bit_true_cosim.py`
- `src/sc_neurocore/neurons/reference_receipts/ermentrout_kopell_1986.json`
- `hdl/formal/catalogue/sc_ermentrout_kopell_map_neuron.sby`
- `hdl/reports/yosys_ermentrout_kopell_map_q88_2026-08-28.json`

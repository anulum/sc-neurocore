<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# AiharaMapNeuron

`AiharaMapNeuron` implements the reduced chaotic neuron described by Aihara's
primary-author manuscript and the subsequent Aihara–Takabe–Toyoda article. It
is a one-state map with a graded logistic output. SC-NeuroCore's original
two-state engineering recurrence remains available as the distinct
[`SCChaoticMapNeuron`](sc_chaotic_map.md), without Aihara paper attribution.

## Source custody

- K. Aihara, “Chaotic Neural Networks,” *RIMS Kokyuroku* 710 (1989),
  145–163, Eqs. 8–12. The manuscript cites the article below as submitted.
- K. Aihara, T. Takabe, and M. Toyoda, “Chaotic neural networks,” *Physics
  Letters A* 144(6–7), 333–340 (1990),
  DOI [`10.1016/0375-9601(90)90136-C`](https://doi.org/10.1016/0375-9601(90)90136-C).

The implementation is anchored to the accessible primary-author equations,
not to equations inferred from secondary descriptions.

## Equations

With identity refractory output `g(x)=x`, the reduced map is

\[
y_{t+1}=k y_t-\alpha x_t+a+I_t,
\qquad
x_t=f(y_t)=\frac{1}{1+\exp(-y_t/\epsilon)}.
\]

`y` is the only independent state. `x` is a derived graded output. `bias`
stores the constant effective stimulus `a`; the API's `current` is an
additional effective stimulus. A raw historical input `A(t)` is not equivalent
to `current`: Aihara's Eq. 5 first transforms it using `A(t-1)`.

The binary observable is also source-defined. Equation 12 gives

\[
h(x)=\begin{cases}1,&x\ge 0.5\\0,&x<0.5.\end{cases}
\]

It is a level waveform shaper, not an upward-crossing detector. Since the
logistic is monotone, the post-update event is equivalently `y_next >= 0`.

## Source-anchored defaults

| Field | Default | Provenance |
|---|---:|---|
| `y` | `0.1` | Figure 5 initial condition |
| `k` | `0.7` | Figure 4 examples |
| `alpha` | `1.0` | Figure 5 sweep fixes the refractory scale to one |
| `bias` | `0.3968` | Figure 4 chaotic example |
| `epsilon` | `0.01` | Figure 4 examples |

The Figure 4 periodic comparison uses `bias=0.6288` with the same `k` and
`epsilon`. Figure 5 separately uses `k=0.6`, `alpha=1`, `epsilon=0.015`, and
`y0=0.1`.

## Python API

```python
import numpy as np

from sc_neurocore.neurons.models import AiharaMapNeuron

neuron = AiharaMapNeuron()
event = neuron.step(0.0)
print(neuron.y, neuron.x, event)

receipt = neuron.simulate(
    0.04 * np.sin(np.arange(512) * 0.037),
    backend="auto",
)
print(receipt["y_final"], receipt["spike_count"])
```

`simulate` returns complete `y`, `x`, and `spikes` arrays plus `y_final`,
`x_final`, and `spike_count`. Inputs and configuration are validated before
native output is accepted. Explicit unavailable backends fail closed; only
`auto` may select another lane.

## Polyglot execution

The maintained lanes all implement the same recurrence and receipts:

| Lane | Surface |
|---|---|
| Python | `sc_neurocore.accel.aihara_map` golden dispatcher |
| Rust | `engine/src/neurons/aihara_map.rs` and dedicated PyO3 binding |
| Julia | dedicated `.jl` kernel and Python facade |
| Go | checked C-shared ABI and service mirror |
| Mojo | checked C ABI with validation-before-write |

The recurrence is chaotic, so last-bit differences in a transcendental
implementation amplify with horizon. Verification therefore separates two
claims:

- tight short-horizon equation parity (`5e-11` over 64 maintained steps);
- a measured 512-step Mojo state envelope (`2e-4`) with exact Eq. 12 events.

This is stronger and more honest than claiming indefinite pointwise identity
for chaotic trajectories.

## Schema, reference trace, and RTL

Paired TOML/JSON schemas encode the one-state map and level event. The committed
`aihara_map_primary.json` oracle is independently iterated from Eqs. 10–12 and
pins the source operating point.

The equation compiler emits `sc_aihara_map.v` in Q8.24. Its logistic LUT needs
explicit signed boundary casts for negative arguments; a regression test pins
those casts. Bounded co-simulation shows exact event decisions and `y` error
below `0.01` for the first 12 autonomous chaotic steps. The bounded horizon is
intentional: quantised chaotic trajectories decorrelate and must not be sold as
long-horizon state equivalence.

The accompanying SymbiYosys job proves reset hygiene and the public Eq. 12
relation `spike_out == !y_out[31]` after active updates. Yosys compilation and
the depth-6 Z3 BMC are part of the focused Model 43 evidence.

## Validation surfaces

- `tests/test_model_aihara_map_neuron.py`: scalar contracts and atomic errors.
- `tests/test_reference_aihara_map.py`: independent primary-equation oracle.
- `tests/test_aihara_map_backends.py`: full native receipts and parity bounds.
- `tests/test_cosim_aihara_map.py`: paired-schema and Q8.24 bounded co-sim.
- Rust and Go unit tests: source first-step, Eq. 12 level semantics, and
  no-mutation failure behavior.

## Scope boundary

The model exposes a single-neuron reduced map. Network coupling belongs in the
network layer, where the effective stimulus may include weighted graded outputs
from other units. The implementation does not fabricate a second recovery
state, clamp the scientific state, reset non-finite candidates, or reinterpret
the source level waveform as an edge event.

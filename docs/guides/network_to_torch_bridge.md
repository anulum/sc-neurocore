<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
# SC-NeuroCore — Explicit bridge from declarative Network to differentiable torch execution

# Network To Torch Bridge

`Network.to_torch()` adds an explicit differentiable path for bounded
declarative network graphs without altering the existing NumPy or Rust
execution paths.

## What it does

```python
bridge = network.to_torch()
counts = bridge(inputs)
```

The returned module:

- accepts input currents with shape `(T, batch, input_dim)`
- keeps the declarative population/projection graph structure
- uses the same previous-spike projection semantics as the NumPy backend
- exposes trainable projection weights through PyTorch parameters
- can write learned weights back with `bridge.sync_to_network()`

## Current support boundary

Supported population models, each stepped as the Python model steps it, with
the model's `v >= v_threshold` comparison and a hard reset to `v_reset`:

- `SCLapicqueLIFNeuron`, or `LapicqueNeuron` in its SC LIF profile (exact flow
  `v_inf + (v - v_inf)·e^(-dt/tau)`); the 1907 source profile is refused
- `StochasticLIFNeuron` (forward Euler) only when:
  - `noise_std == 0.0`
  - `refractory_period == 0`
  - `entropy_source is None`

Every neuron of a population must share its parameters; each starts from its
own voltage and, for StochasticLIF, its own live refractory counter (a neuron
whose counter is set through its state holds at `v_rest` for that many steps,
as `step()` does). A StochasticLIF step whose candidate overflows to +infinity
still spikes and resets to `v_reset`, as the Python neuron does.

The models' runtime refusals are kept: a step refuses a non-finite voltage
(StochasticLIF refuses the step after a candidate that overflowed negatively;
the last step of a run may end there, as in the model), and the SC Lapicque
flow refuses a non-finite `v_inf` or candidate in the step that forms it, and a
voltage at or above threshold. These raise `ValueError` with the model's message.

The step is evaluated in float64, the precision of the Python models, whatever
the floating dtype of the input: a float32 input is widened exactly and gives the
events and voltages the model gives for those float32 values (no float32
overflow or rounding of `R`, `I` or `v`). Counts and traces are returned in the
input's dtype; a gradient larger than that dtype holds rounds to infinity there.

Not supported yet:

- embedded `TimedArray`, `PoissonInput`, or `StepCurrent`
- closed recurrent graphs with no population that can receive external input
- plastic projections
- delayed projections
- population models whose state updates do not map cleanly to the current
  differentiable cell library

Unsupported cases raise `NotImplementedError` explicitly. The bridge does not
silently approximate them.

## Why the boundary is narrow

The goal is to reconnect the declarative `Network` API to differentiable
training without pretending that every neuron model in the repository already
has a faithful training-cell equivalent.

That is safer than forcing a broad lossy conversion.

## Weight semantics

The bridge materialises each projection as a masked dense parameter tensor.
Only edges present in the original CSR graph remain trainable; absent edges are
masked to zero.

`sync_to_network()` writes the learned values back into the original CSR data.

The weight parameters are float64, the precision of the CSR data, so a sync
without training writes back every weight unchanged and a float64 run uses the
network's own weights; the forward pass uses them in float64 for any input
dtype. Converting the bridge itself (for example `bridge.float()`) rounds the stored weights, and a
later sync writes those rounded values. Parallel CSR entries (one source
reaching one target twice) and a nonzero `weight_threshold` are refused.

## Example

```python
import torch

from sc_neurocore.network import Network, Population, Projection
from sc_neurocore.training.surrogate import atan_surrogate_custom_op

src = Population("LapicqueNeuron", 2, params={"tau": 5.0, "dt": 1.0}, label="src")
tgt = Population("LapicqueNeuron", 2, params={"tau": 5.0, "dt": 1.0}, label="tgt")
proj = Projection(src, tgt, weight=0.8, probability=1.0, seed=42)
net = Network(src, tgt, proj)

bridge = net.to_torch(surrogate_fn=atan_surrogate_custom_op)
inputs = torch.zeros(8, 4, 2)
counts = bridge(inputs)
bridge.sync_to_network()
```

## Output convention

The forward path returns spike counts for output populations, where output
means populations with zero outgoing projections. If every population has an
outgoing projection, the bridge falls back to returning counts for all
populations.

The input surface is stricter: at least one population must have no incoming
projection so the `(T, batch, input_dim)` current tensor has a deterministic
target. A closed recurrent graph must be wrapped with an explicit input-facing
population before it can be converted.

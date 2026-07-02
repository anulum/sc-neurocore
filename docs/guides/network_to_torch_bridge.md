# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
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

Supported population models:

- `LapicqueNeuron`
- `StochasticLIFNeuron` only when:
  - `noise_std == 0.0`
  - `refractory_period == 0`
  - `entropy_source is None`
  - `v_reset == v_rest`

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

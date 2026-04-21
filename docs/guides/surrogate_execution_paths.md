# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Explicit execution paths for surrogate gradients

# Surrogate Execution Paths

SC-NeuroCore now keeps two explicit execution paths for the six PyTorch
surrogate gradients:

- `custom_op`
- `legacy_autograd`

This is deliberate. The old path remains available for comparison and
regression testing, while the modern path is available for `torch.compile`
and the current PyTorch operator stack.

## Available paths

The module-level constant is:

```python
from sc_neurocore.training.surrogate import SURROGATE_PATHS

assert SURROGATE_PATHS == ("custom_op", "legacy_autograd")
```

Each surrogate now has three public call forms:

- default dispatcher, for example `fast_sigmoid(...)`
- explicit modern path, for example `fast_sigmoid_custom_op(...)`
- explicit legacy path, for example `fast_sigmoid_legacy(...)`

The same pattern exists for:

- `superspike`
- `atan_surrogate`
- `sigmoid_surrogate`
- `straight_through`
- `triangular`

## Default behaviour

The public dispatcher functions default to `path="custom_op"`.

Example:

```python
from sc_neurocore.training import fast_sigmoid

spike = fast_sigmoid(v_minus_threshold)
```

Explicit legacy comparison:

```python
from sc_neurocore.training.surrogate import fast_sigmoid_legacy

spike = fast_sigmoid_legacy(v_minus_threshold)
```

## LIFCell wiring

`LIFCell` and the other differentiable cell modules still accept any surrogate
callable. That means you can wire the path explicitly:

```python
from sc_neurocore.training import LIFCell
from sc_neurocore.training.surrogate import (
    atan_surrogate_custom_op,
    atan_surrogate_legacy,
)

cell_modern = LIFCell(surrogate_fn=atan_surrogate_custom_op)
cell_legacy = LIFCell(surrogate_fn=atan_surrogate_legacy)
```

## Why keep both?

- the legacy path is the known historical baseline
- the custom-op path is the modern PyTorch integration point
- direct parity tests can catch behavioural drift immediately

This is safer than replacing the old implementation silently and then trying
to infer later whether a regression came from the surrogate itself or from the
execution substrate around it.

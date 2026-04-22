<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Explicit JAX surrogate execution paths -->

# JAX Surrogate Execution Paths

SC-NeuroCore now exposes two explicit JAX surrogate routes for spiking
training. The default path is modern `jax.custom_vjp`. The historical
`stop_gradient` route remains available for parity checks and controlled
comparison.

## Available Paths

### `custom_vjp`

- Hard spikes in the forward pass
- Fast-sigmoid proxy gradient in the backward pass
- Implemented with `jax.custom_vjp`
- Intended default for new JAX training work

### `legacy_stop_gradient`

- Preserves the earlier straight-through reset pattern
- Uses surrogate-valued spike accumulation plus `jax.lax.stop_gradient`
- Kept only so the old training route can still be tested explicitly

## Why The Split Exists

The old JAX code used one implicit path. That made the training behaviour
harder to reason about, harder to compare, and impossible to audit against
the `jax.custom_vjp` requirement without replacing the legacy route.

The new structure keeps both paths side by side:

- no hidden behavioural switch
- no silent replacement of older training code
- direct testability of both routes

## Public API

```python
from sc_neurocore.accel.jax_backend import (
    JAX_SURROGATE_PATHS,
    jax_surrogate_loss,
    jax_surrogate_gradient_step,
)
```

`JAX_SURROGATE_PATHS` is:

```python
("custom_vjp", "legacy_stop_gradient")
```

## Example

```python
import jax.numpy as jnp
from sc_neurocore.accel.jax_backend import jax_surrogate_gradient_step

weights = [jnp.asarray([[0.4, -0.1], [0.2, 0.5]], dtype=jnp.float32)]
x = jnp.asarray([[1.0, 0.2], [0.3, 1.1]], dtype=jnp.float32)
targets = jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)

updated, loss_value = jax_surrogate_gradient_step(
    weights,
    x,
    targets,
    n_steps=5,
    lr=1e-2,
    surrogate_path="custom_vjp",
)
```

To compare against the historical route:

```python
updated_legacy, loss_legacy = jax_surrogate_gradient_step(
    weights,
    x,
    targets,
    n_steps=5,
    lr=1e-2,
    surrogate_path="legacy_stop_gradient",
)
```

## Verification Boundary

The `custom_vjp` path is verified against:

- a NumPy finite-difference check of the fast-sigmoid proxy gradient
- `jax.jit(jax.vmap(jax.grad(...)))` on a small training loss

If JAX is not installed, the JAX-only regression file is skipped and the
fallback tests still verify the exported path declarations.

## Recommendation

Use `custom_vjp` for new JAX training experiments. Keep
`legacy_stop_gradient` only when you need before/after comparison against
older runs.

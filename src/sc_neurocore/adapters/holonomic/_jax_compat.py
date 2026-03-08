# SPDX-License-Identifier: AGPL-3.0-or-later
"""JAX/NumPy compatibility layer for holonomic adapters."""

from __future__ import annotations

import numpy as np

__all__ = ["HAS_JAX", "jnp", "make_rng", "maybe_jit", "normal", "split_rng", "uniform"]

try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    jax = None  # type: ignore[assignment]
    import numpy as jnp  # type: ignore[no-redef]

    HAS_JAX = False


def make_rng(seed: int):
    """JAX PRNGKey when available, else NumPy Generator."""
    if HAS_JAX:
        return jax.random.PRNGKey(seed)
    return np.random.default_rng(seed)


def split_rng(key):
    """Split PRNG: JAX functional split or NumPy stateful pass-through."""
    if HAS_JAX:
        return jax.random.split(key)
    return key, key


def uniform(key, shape, minval=0.0, maxval=1.0):
    """Uniform samples from JAX or NumPy PRNG."""
    if HAS_JAX:
        return jax.random.uniform(key, shape, minval=minval, maxval=maxval)
    return np.asarray(key.random(shape) * (maxval - minval) + minval)


def normal(key, shape):
    """Standard normal samples from JAX or NumPy PRNG."""
    if HAS_JAX:
        return jax.random.normal(key, shape)
    return np.asarray(key.standard_normal(shape))


def maybe_jit(fn):
    """@jax.jit when available, identity otherwise."""
    if HAS_JAX:
        return jax.jit(fn)
    return fn

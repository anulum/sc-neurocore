# SPDX-License-Identifier: AGPL-3.0-or-later
"""JAX compatibility helpers for holonomic adapters.

Thin wrappers around JAX RNG and JIT that fall back to NumPy when JAX
is unavailable, so adapter code can stay backend-agnostic.
"""
from __future__ import annotations

import numpy as np

try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False


def make_rng(seed: int = 0):
    """Create a PRNG key (JAX) or seed array (NumPy fallback)."""
    if HAS_JAX:
        return jax.random.PRNGKey(seed)
    return np.array([0, seed], dtype=np.uint32)


def split_rng(key):
    """Split a PRNG key into two children."""
    if HAS_JAX:
        return jax.random.split(key)
    s = int(key[-1])
    return np.array([0, s + 1], dtype=np.uint32), np.array([0, s + 2], dtype=np.uint32)


def uniform(key, shape: tuple):
    """Uniform [0, 1) samples."""
    if HAS_JAX:
        return jax.random.uniform(key, shape)
    rng = np.random.default_rng(int(key[-1]))
    return rng.uniform(size=shape).astype(np.float32)


def normal(key, shape: tuple):
    """Standard normal samples."""
    if HAS_JAX:
        return jax.random.normal(key, shape)
    rng = np.random.default_rng(int(key[-1]))
    return rng.standard_normal(size=shape).astype(np.float32)


def maybe_jit(fn):
    """JIT-compile if JAX available, otherwise identity."""
    if HAS_JAX:
        return jax.jit(fn)
    return fn

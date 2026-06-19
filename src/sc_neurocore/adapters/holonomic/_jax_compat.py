# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — JAX compatibility helpers for holonomic adapters

"""JAX compatibility helpers for holonomic adapters.

Thin wrappers around JAX RNG and JIT that fall back to NumPy when JAX
is unavailable, so adapter code can stay backend-agnostic.
"""

from __future__ import annotations

import types
from collections.abc import Callable
from typing import Any, cast

import numpy as np

try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    jnp: types.ModuleType = np  # type: ignore[no-redef]
    HAS_JAX = False

__all__ = ["jnp", "HAS_JAX", "make_rng", "split_rng", "uniform", "normal", "maybe_jit"]


def make_rng(seed: int = 0) -> np.ndarray[Any, Any]:
    """Create a PRNG key (JAX) or seed array (NumPy fallback)."""
    if HAS_JAX:
        return cast(np.ndarray[Any, Any], jax.random.PRNGKey(seed))
    return np.array([0, seed], dtype=np.uint32)


def split_rng(key: Any) -> tuple[Any, Any]:
    """Split a PRNG key into two children."""
    if HAS_JAX:
        return cast(tuple[Any, Any], jax.random.split(key))
    s = int(key[-1])
    return np.array([0, s + 1], dtype=np.uint32), np.array([0, s + 2], dtype=np.uint32)


def uniform(key: Any, shape: tuple[int, ...], minval: float = 0.0, maxval: float = 1.0) -> Any:
    """Uniform samples in [minval, maxval)."""
    if HAS_JAX:
        return jax.random.uniform(key, shape, minval=minval, maxval=maxval)
    rng = np.random.default_rng(int(key[-1]))
    return rng.uniform(low=minval, high=maxval, size=shape).astype(np.float32)


def normal(key: Any, shape: tuple[int, ...]) -> Any:
    """Standard normal samples."""
    if HAS_JAX:
        return jax.random.normal(key, shape)
    rng = np.random.default_rng(int(key[-1]))
    return rng.standard_normal(size=shape).astype(np.float32)


def maybe_jit(fn: Callable[..., Any]) -> Callable[..., Any]:
    """JIT-compile if JAX available, otherwise identity."""
    if HAS_JAX:
        return cast(Callable[..., Any], jax.jit(fn))
    return fn

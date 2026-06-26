# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Stable acceleration-backend selector for SC inference

"""Stable acceleration-backend selector for the public SC inference surface.

``get_backend`` returns the fastest available backend handle (Rust accelerated
path first, NumPy fallback last). Each handle exposes ``sc_forward`` with the same
contract; the Rust path and the NumPy fallback are bit-identical for a fixed seed.

This replaces the removed ``sc_neurocore.accel.get_backend`` import relied on by
the SCPN-CONTROL Petri-net compiler fast path.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from .sc_inference import sc_forward_numpy

#: Accelerator backends in fastest-measured-first order (numpy is the floor).
ACCELERATORS = ("rust", "mojo", "julia", "go")
#: Full probe order including the always-available numpy fallback.
PRIORITY = (*ACCELERATORS, "numpy")


class Backend(ABC):
    """Acceleration backend handle exposing the stable SC inference contract."""

    #: Stable backend identifier.
    name: str

    @abstractmethod
    def sc_forward(
        self,
        weights_packed: npt.ArrayLike,
        input_probs: npt.ArrayLike,
        *,
        length: int,
        seed: int = 0xACE1,
    ) -> npt.NDArray[np.float64]:
        """Run the stochastic forward pass; see :func:`sc_neurocore.accel.sc_forward`."""


class NumpyBackend(Backend):
    """NumPy fallback backend (always available, the bit-true floor)."""

    name = "numpy"

    def sc_forward(
        self,
        weights_packed: npt.ArrayLike,
        input_probs: npt.ArrayLike,
        *,
        length: int,
        seed: int = 0xACE1,
    ) -> npt.NDArray[np.float64]:
        """Run the NumPy bit-true stochastic forward pass."""
        return sc_forward_numpy(weights_packed, input_probs, length, seed)


class RustBackend(Backend):
    """Rust-accelerated backend over the compiled engine."""

    name = "rust"

    def sc_forward(
        self,
        weights_packed: npt.ArrayLike,
        input_probs: npt.ArrayLike,
        *,
        length: int,
        seed: int = 0xACE1,
    ) -> npt.NDArray[np.float64]:
        """Run the compiled Rust stochastic forward pass."""
        from sc_neurocore_engine import py_sc_forward_packed

        weights = np.ascontiguousarray(weights_packed, dtype=np.uint64)
        probs = np.ascontiguousarray(input_probs, dtype=np.float64).reshape(-1)
        if weights.ndim != 3:
            raise ValueError(
                f"weights_packed must be 3-D (n_out, n_in, n_words), got {weights.ndim}-D"
            )
        n_out, n_in, n_words = (int(dim) for dim in weights.shape)
        result = py_sc_forward_packed(
            weights.reshape(-1), n_out, n_in, n_words, probs, int(length), int(seed)
        )
        return np.ascontiguousarray(result, dtype=np.float64)


def _probe(name: str) -> Backend | None:
    """Return a working backend handle for ``name``, or ``None`` if unavailable."""
    if name == "numpy":
        return NumpyBackend()
    if name == "rust":
        candidate = RustBackend()
        try:
            candidate.sc_forward(
                np.zeros((1, 1, 1), dtype=np.uint64), np.zeros(1, dtype=np.float64), length=1
            )
        except (ImportError, OSError, RuntimeError, ValueError):
            return None
        return candidate
    return None


def available_backends() -> dict[str, bool]:
    """Report which SC inference backends resolve, in fastest-first order.

    Returns
    -------
    dict
        Mapping of backend name to availability; ``numpy`` is always ``True``.
    """
    return {name: _probe(name) is not None for name in PRIORITY}


def get_backend(name: str = "auto") -> Backend:
    """Return an SC inference backend handle.

    Parameters
    ----------
    name : str, optional
        ``"auto"`` (default) returns the fastest available backend in
        :data:`PRIORITY` order; a specific name (``"rust"``, ``"numpy"``) forces
        that backend.

    Returns
    -------
    Backend
        A handle whose ``sc_forward`` matches the documented contract.

    Raises
    ------
    ValueError
        If ``name`` is not ``"auto"`` or a known backend name.
    RuntimeError
        If an explicitly requested backend is unavailable.
    """
    if name == "auto":
        for candidate in ACCELERATORS:
            backend = _probe(candidate)
            if backend is not None:
                return backend
        return NumpyBackend()
    if name not in PRIORITY:
        raise ValueError(f"unknown backend {name!r}; choose from {('auto', *PRIORITY)}")
    backend = _probe(name)
    if backend is None:
        raise RuntimeError(f"backend {name!r} is not available in this environment")
    return backend


__all__ = ["Backend", "NumpyBackend", "RustBackend", "available_backends", "get_backend"]

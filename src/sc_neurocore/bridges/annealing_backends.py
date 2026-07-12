# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Optional quantum-annealing backend adapters

"""Isolate optional D-Wave, dimod, and native-engine dependencies.

The public bridge remains usable without any optional backend. Callers that
explicitly request a missing backend receive a stable runtime error instead of
silently running a different implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast


RustEnergyKernel = Callable[
    [
        Sequence[int],
        Sequence[float],
        Sequence[int],
        Sequence[int],
        Sequence[float],
        Sequence[int],
        float,
    ],
    float,
]
RustBatchEnergyKernel = Callable[
    [
        Sequence[int],
        Sequence[float],
        Sequence[int],
        Sequence[int],
        Sequence[float],
        Sequence[Sequence[int]],
        float,
    ],
    Sequence[float],
]
RustAnnealingKernel = Callable[..., Mapping[str, object]]


try:
    import dimod as _dimod_module
except ImportError:
    dimod: Any | None = None
    HAS_DIMOD = False
else:
    dimod = _dimod_module
    HAS_DIMOD = True

try:
    from dwave.system import DWaveSampler as _DWaveSampler
    from dwave.system import EmbeddingComposite as _EmbeddingComposite
except ImportError:
    DWaveSampler: Any | None = None
    EmbeddingComposite: Any | None = None
    HAS_DWAVE = False
else:
    DWaveSampler = _DWaveSampler
    EmbeddingComposite = _EmbeddingComposite
    HAS_DWAVE = True


_rust_ising_energy: RustEnergyKernel | None = None
_rust_batch_energy: RustBatchEnergyKernel | None = None
_rust_simulated_annealing: RustAnnealingKernel | None = None

try:
    from sc_neurocore_engine.quantum import (
        get_batch_ising_energy,
        get_ising_energy,
        get_simulated_annealing,
        has_full_quantum_annealing_backend,
    )

    _rust_ising_energy = cast(RustEnergyKernel, get_ising_energy())
    _rust_batch_energy = cast(RustBatchEnergyKernel, get_batch_ising_energy())
    _rust_simulated_annealing = cast(RustAnnealingKernel, get_simulated_annealing())
    HAS_RUST_QA = bool(has_full_quantum_annealing_backend())
except (ImportError, RuntimeError):
    HAS_RUST_QA = False

if not all(
    callable(kernel)
    for kernel in (_rust_ising_energy, _rust_batch_energy, _rust_simulated_annealing)
):
    HAS_RUST_QA = False


def require_rust_energy() -> RustEnergyKernel:
    """Return the native energy kernel or raise a stable availability error."""
    if not HAS_RUST_QA or _rust_ising_energy is None:
        raise RuntimeError("Rust quantum-annealing energy backend is unavailable")
    return _rust_ising_energy


def require_rust_batch_energy() -> RustBatchEnergyKernel:
    """Return the native batch kernel or raise a stable availability error."""
    if not HAS_RUST_QA or _rust_batch_energy is None:
        raise RuntimeError("Rust quantum-annealing batch backend is unavailable")
    return _rust_batch_energy


def require_rust_annealer() -> RustAnnealingKernel:
    """Return the native annealer or raise a stable availability error."""
    if not HAS_RUST_QA or _rust_simulated_annealing is None:
        raise RuntimeError("Rust quantum-annealing solver backend is unavailable")
    return _rust_simulated_annealing


def build_spin_bqm(
    h: Mapping[int, float],
    couplings: Mapping[tuple[int, int], float],
    offset: float,
) -> Any | None:
    """Build a dimod spin BQM, returning ``None`` when dimod is absent."""
    if not HAS_DIMOD or dimod is None:
        return None
    return dimod.BinaryQuadraticModel(h, couplings, offset, "SPIN")


def require_dwave_components() -> tuple[Any, Any, Any]:
    """Return dimod and D-Wave constructors or raise an availability error."""
    if (
        not HAS_DIMOD
        or not HAS_DWAVE
        or dimod is None
        or DWaveSampler is None
        or EmbeddingComposite is None
    ):
        raise RuntimeError("D-Wave Ocean SDK and dimod are required for QPU submission")
    return dimod, DWaveSampler, EmbeddingComposite

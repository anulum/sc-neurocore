# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_quadratic_if_backends.py

from __future__ import annotations

"""End-to-end parity and rejection contracts for every native QIF lane."""

import ctypes


import math


import subprocess


from collections.abc import Callable


from pathlib import Path


from typing import cast


from unittest.mock import patch


import numpy as np


import numpy.typing as npt


import pytest


from sc_neurocore.accel import quadratic_if as backends


from sc_neurocore.neurons.models.quadratic_if import QuadraticIFNeuron


_REPOSITORY = Path(__file__).resolve().parents[1]


_GOLDENS = (
    (0.0, 0),
    (0.333, 2),
    (0.5, 3),
    (1.0, 6),
    (2.0, 11),
    (5.0, 26),
    (20.0, 100),
    (50.0, 250),
)


_COMPILED_BACKENDS = ("rust", "julia", "go", "mojo")


_TRACE_ATOL = 2.0e-12


def _run(
    backend: str,
    *,
    current: float,
    n_steps: int = 1_000,
    factory: Callable[[], QuadraticIFNeuron] = QuadraticIFNeuron,
) -> tuple[npt.NDArray[np.float64], int, float]:
    """Run one backend and return its trace, event count, and final state."""
    neuron = factory()
    trace, spikes = neuron.simulate(n_steps, current, backend=backend)
    return trace, spikes, neuron.v


def _configured() -> QuadraticIFNeuron:
    """Return a non-default state exercising the complete native ABI."""
    return QuadraticIFNeuron(v=-0.37, v_reset=-1.3, v_peak=1.7, dt=0.037)


def _c_arguments(neuron: QuadraticIFNeuron) -> tuple[float, ...]:
    """Return numeric fields in the C-ABI declaration order."""
    return (neuron.v, neuron.v_reset, neuron.v_peak, neuron.dt)


def _require_qif_backend(name: str) -> None:
    """Load a real compiled QIF lane or skip when it is not built."""
    loaders = {
        "julia": backends.ensure_julia_loaded,
        "go": backends.ensure_go_loaded,
        "mojo": backends.ensure_mojo_loaded,
    }
    if name not in loaders:
        raise ValueError(f"unsupported QIF backend name: {name!r}")
    if not loaders[name]():
        pytest.skip(f"{name} QIF backend is not built in this environment")


__all__ = [
    "ctypes",
    "math",
    "subprocess",
    "Callable",
    "Path",
    "cast",
    "patch",
    "np",
    "npt",
    "pytest",
    "backends",
    "QuadraticIFNeuron",
    "_REPOSITORY",
    "_GOLDENS",
    "_COMPILED_BACKENDS",
    "_TRACE_ATOL",
    "_run",
    "_configured",
    "_c_arguments",
    "_require_qif_backend",
]

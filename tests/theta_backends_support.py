# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_theta_backends.py

from __future__ import annotations

"""End-to-end parity and rejection contracts for every native Theta lane."""

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

from sc_neurocore.accel import theta as backends

from sc_neurocore.neurons.models.theta import ThetaNeuron

_REPOSITORY = Path(__file__).resolve().parents[1]

_GOLDENS = (
    (-1.0, 0),
    (-0.5, 0),
    (0.0, 0),
    (0.1, 1),
    (0.333, 2),
    (0.5, 2),
    (1.0, 3),
    (2.0, 5),
    (5.0, 7),
    (20.0, 14),
    (50.0, 23),
)

_COMPILED_BACKENDS = ("rust", "julia", "go", "mojo")

_PHASE_ATOL = 2.0e-12

def _run(
    backend: str,
    *,
    current: float,
    n_steps: int = 1_000,
    factory: Callable[[], ThetaNeuron] = ThetaNeuron,
) -> tuple[npt.NDArray[np.float64], int, float]:
    """Run one backend and return its trace, event count, and final phase."""
    neuron = factory()
    trace, spikes = neuron.simulate(n_steps, current, backend=backend)
    return trace, spikes, neuron.theta

def _configured() -> ThetaNeuron:
    """Return a non-default state exercising the complete native ABI."""
    return ThetaNeuron(theta=0.37, dt=0.037)

def _phase_delta(
    actual: npt.ArrayLike,
    expected: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Return the shortest signed distance between phases on the circle."""
    actual_array = np.asarray(actual, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    return (actual_array - expected_array + math.pi) % (2.0 * math.pi) - math.pi

def _assert_phase_parity(actual: npt.ArrayLike, expected: npt.ArrayLike) -> None:
    """Enforce the measured cross-libm phase envelope modulo two pi."""
    delta = _phase_delta(actual, expected)
    np.testing.assert_allclose(delta, np.zeros_like(delta), rtol=0.0, atol=_PHASE_ATOL)


__all__ = ['ctypes', 'math', 'subprocess', 'Callable', 'Path', 'cast', 'patch', 'np', 'npt', 'pytest', 'backends', 'ThetaNeuron', '_REPOSITORY', '_GOLDENS', '_COMPILED_BACKENDS', '_PHASE_ATOL', '_run', '_configured', '_phase_delta', '_assert_phase_parity']

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_dpi_neuron_backends.py

from __future__ import annotations

"""End-to-end parity and rejection contracts for every native DPI lane."""

from collections.abc import Callable


import ctypes


import math


import os


from pathlib import Path


import subprocess


import sys


from unittest.mock import patch


import numpy as np


import numpy.typing as npt


import pytest


from sc_neurocore.accel import dpi_neuron as backends


from sc_neurocore.neurons.models.dpi_neuron import DPINeuron


_REPOSITORY = Path(__file__).resolve().parents[1]


_GOLDENS = (
    (-0.1, 0),
    (0.0, 0),
    (1.0, 0),
    (2.0, 0),
    (3.0, 1),
    (5.0, 3),
    (10.0, 6),
    (20.0, 11),
    (50.0, 21),
)


_FULL_CONTRACT_BACKENDS = ("julia", "go", "mojo")


_COMPILED_BACKENDS = ("rust", *_FULL_CONTRACT_BACKENDS)


_STATE_ATOL = 5.0e-13


def _configured() -> DPINeuron:
    """Return a stable non-default state exercising the complete native ABI."""
    return DPINeuron(
        i_mem=0.37,
        i_ahp=0.08,
        refractory_time=0.0,
        i_threshold=1.3,
        i_reset=0.2,
        i_rest=0.15,
        i_tau=0.9,
        i_g=1.4,
        i_tau_ahp=0.12,
        i_ga=0.8,
        i_spike=4.2,
        i_0=0.02,
        kappa=0.65,
        alpha=8.0,
        tau=7.0,
        tau_ahp=45.0,
        refractory_period=0.6,
        dt=0.05,
    )


def _factory_values() -> tuple[float, ...]:
    """Return the 18-double native ABI prefix in public-model order."""
    neuron = DPINeuron()
    return (
        neuron.i_mem,
        neuron.i_ahp,
        neuron.refractory_time,
        neuron.i_threshold,
        neuron.i_reset,
        neuron.i_rest,
        neuron.i_tau,
        neuron.i_g,
        neuron.i_tau_ahp,
        neuron.i_ga,
        neuron.i_spike,
        neuron.i_0,
        neuron.kappa,
        neuron.alpha,
        neuron.tau,
        neuron.tau_ahp,
        neuron.refractory_period,
        neuron.dt,
    )


def _run(
    backend: str,
    *,
    current: float,
    n_steps: int = 1_000,
    configured: bool = False,
) -> tuple[npt.NDArray[np.float64], int, tuple[float, float, float]]:
    """Run one backend and return its trace, events, and all final states."""
    neuron = _configured() if configured else DPINeuron()
    trace, spikes = neuron.simulate(n_steps, current, backend=backend)
    return trace, spikes, (neuron.i_mem, neuron.i_ahp, neuron.refractory_time)


def _run_complete(
    backend: str,
    *,
    current: float,
    n_steps: int = 1_000,
    configured: bool = False,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.uint8],
    tuple[float, float, float],
]:
    """Run one public complete packet and return its committed final state."""
    neuron = _configured() if configured else DPINeuron()
    i_mem, i_ahp, refractory, events = neuron.simulate_complete(n_steps, current, backend=backend)
    state = (neuron.i_mem, neuron.i_ahp, neuron.refractory_time)
    return i_mem, i_ahp, refractory, events, state


def _assert_state_parity(actual: npt.ArrayLike, expected: npt.ArrayLike) -> None:
    """Enforce the measured cross-runtime floating-point envelope."""
    actual_array = np.asarray(actual, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    np.testing.assert_allclose(actual_array, expected_array, rtol=0.0, atol=_STATE_ATOL)


def _invoke_full_contract(runner: Callable[..., object]) -> object:
    """Invoke one configurable native runner with the public 20-field ABI."""
    return runner(*_factory_values(), 1, 0.0)


__all__ = [
    "Callable",
    "ctypes",
    "math",
    "os",
    "Path",
    "subprocess",
    "sys",
    "patch",
    "np",
    "npt",
    "pytest",
    "backends",
    "DPINeuron",
    "_REPOSITORY",
    "_GOLDENS",
    "_FULL_CONTRACT_BACKENDS",
    "_COMPILED_BACKENDS",
    "_STATE_ATOL",
    "_configured",
    "_factory_values",
    "_run",
    "_run_complete",
    "_assert_state_parity",
    "_invoke_full_contract",
]

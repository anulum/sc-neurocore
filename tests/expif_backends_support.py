# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_expif_backends.py

from __future__ import annotations

"""End-to-end parity and rejection contracts for every ExpIF native lane."""

import ctypes


import importlib


import math


import os


from collections.abc import Callable


from typing import cast


import numpy as np


import numpy.typing as npt


import pytest


import sc_neurocore.neurons.models.expif as expif


from sc_neurocore.neurons.models.expif import ExpIFNeuron


_TRACE_ATOL = 5.0e-8


_GOLDENS = ((0.0, 0), (5.0, 0), (20.0, 2), (50.0, 5))


_COMPILED_BACKENDS = ("rust", "julia", "go", "mojo")


def _run(
    backend: str,
    *,
    current: float,
    n_steps: int = 1_000,
    factory: Callable[[], ExpIFNeuron] = ExpIFNeuron,
) -> tuple[npt.NDArray[np.float64], int, tuple[float, float]]:
    """Run one backend and return its trace, event count, and final state."""
    neuron = factory()
    trace, spikes = neuron.simulate(n_steps, current, backend=backend)
    return trace, spikes, (neuron.v, neuron.refractory_remaining)


def _configured() -> ExpIFNeuron:
    """Return a non-default state that exercises the complete native ABI."""
    return ExpIFNeuron(
        v=-62.0,
        v_rest=-64.0,
        v_reset=-69.0,
        v_threshold=25.0,
        v_rh=-58.0,
        delta_t=3.0,
        tau=12.0,
        dt=0.03,
        refractory_period=0.09,
        refractory_remaining=0.06,
    )


def _c_arguments(neuron: ExpIFNeuron) -> tuple[float, ...]:
    """Return numeric fields in the C-ABI declaration order."""
    return (
        neuron.v,
        neuron.v_rest,
        neuron.v_reset,
        neuron.v_threshold,
        neuron.v_rh,
        neuron.delta_t,
        neuron.tau,
        neuron.dt,
        neuron.refractory_period,
        neuron.refractory_remaining,
    )


def _require_expif_backend(name: str) -> None:
    """Load a real compiled ExpIF lane or skip when it is not built.

    Fall-through auto tests must load the selected backend for real before
    disabling higher-priority ``_ensure_*`` helpers; otherwise
    ``simulate`` asserts on a still-empty module/library handle.
    """
    loaders = {
        "julia": expif._ensure_julia_loaded,
        "go": expif._ensure_go_loaded,
        "mojo": expif._ensure_mojo_loaded,
    }
    if name not in loaders:
        raise ValueError(f"unsupported ExpIF backend name: {name!r}")
    if not loaders[name]():
        pytest.skip(f"{name} ExpIF backend is not built in this environment")



__all__ = ['ctypes', 'importlib', 'math', 'os', 'Callable', 'cast', 'np', 'npt', 'pytest', 'expif', 'ExpIFNeuron', '_TRACE_ATOL', '_GOLDENS', '_COMPILED_BACKENDS', '_run', '_configured', '_c_arguments', '_require_expif_backend']

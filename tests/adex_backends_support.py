# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_adex_backends.py

from __future__ import annotations

"""End-to-end parity and rejection contracts for every AdEx acceleration lane."""

import ctypes


import importlib


import math


import os


from collections.abc import Callable


from typing import Literal, cast


import numpy as np


import numpy.typing as npt


import pytest


import sc_neurocore.neurons.models.adex as adex


from sc_neurocore.neurons.models.adex import AdExNeuron


_TRACE_ATOL = 5.0e-12


_GOLDENS = ((0.0, 0), (200.0, 4), (500.0, 12))


_COMPILED_BACKENDS = ("rust", "julia", "go", "mojo")


def _run(
    backend: str,
    *,
    current: float,
    n_steps: int = 1_000,
    factory: Callable[[], AdExNeuron] = AdExNeuron,
) -> tuple[npt.NDArray[np.float64], int, tuple[float, float]]:
    neuron = factory()
    trace, spikes = neuron.simulate(n_steps, current, backend=backend)
    return trace, spikes, (neuron.v, neuron.w)


def _c_arguments(neuron: AdExNeuron) -> tuple[float, ...]:
    return (
        neuron.v,
        neuron.w,
        neuron.v_rest,
        neuron.v_reset,
        neuron.v_threshold,
        neuron.v_rh,
        neuron.delta_t,
        neuron.tau,
        neuron.tau_w,
        neuron.a,
        neuron.b,
        neuron.c_m,
        neuron.dt,
    )


def _require_adex_backend(name: str) -> None:
    """Load a real compiled AdEx lane or skip when it is not built.

    Auto-dispatch fall-through tests must not monkeypatch ``_ensure_*`` to
    return ``True`` without a loaded handle: ``simulate`` then hits
    ``assert _julia_module is not None`` (and the Go/Mojo equivalents) and
    fails when the suite runs without a prior parity load.
    """
    loaders = {
        "julia": adex._ensure_julia_loaded,
        "go": adex._ensure_go_loaded,
        "mojo": adex._ensure_mojo_loaded,
    }
    if name not in loaders:
        raise ValueError(f"unsupported AdEx backend name: {name!r}")
    if not loaders[name]():
        pytest.skip(f"{name} AdEx backend is not built in this environment")



__all__ = ['ctypes', 'importlib', 'math', 'os', 'Callable', 'Literal', 'cast', 'np', 'npt', 'pytest', 'adex', 'AdExNeuron', '_TRACE_ATOL', '_GOLDENS', '_COMPILED_BACKENDS', '_run', '_c_arguments', '_require_adex_backend']

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rulkov backend test support

"""Shared imports, backend probes, and reference runner for Rulkov parity tests."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from sc_neurocore.neurons.models import rulkov_map
from sc_neurocore.neurons.models.rulkov_map import RulkovMapNeuron

_ULP = float(np.spacing(1.0))
_STEP_TOL = 8.0 * _ULP


def _run(
    backend: str, *, sigma: float = -1.6, n: int = 4000, current: float = 0.5
) -> tuple[NDArray[np.float64], int, float, float]:
    neuron = RulkovMapNeuron(sigma=sigma)
    trace, spikes = neuron.simulate(n, current, backend=backend)
    return trace, spikes, neuron.x, neuron.y


def _rust() -> bool:
    return rulkov_map._HAS_RUST


def _julia() -> bool:
    return rulkov_map._ensure_julia_loaded()


def _go() -> bool:
    return rulkov_map._ensure_go_loaded()


def _mojo() -> bool:
    return rulkov_map._ensure_mojo_loaded()


_BIT_EXACT: list[tuple[str, Callable[[], bool]]] = [
    ("rust", _rust),
    ("julia", _julia),
    ("go", _go),
]
_REGIMES = [-1.6, -0.5, 0.5, 1.0]

__all__ = [
    "RulkovMapNeuron",
    "Callable",
    "_BIT_EXACT",
    "_REGIMES",
    "_STEP_TOL",
    "_go",
    "_julia",
    "_mojo",
    "_run",
    "_rust",
    "np",
    "pytest",
]

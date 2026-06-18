# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Demo routes for the safe alternative-path harness

"""Demonstration routes for the safe alternative-path validation harness.

Registers small, self-contained baseline/candidate route pairs that exercise
the alternative-path registry without depending on external hardware.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .alternative_path import AlternativePathRegistry, AlternativePathRoute


def _baseline_affine_sigmoid(
    x: np.ndarray[Any, Any] | list[float], bias: float = 0.0
) -> np.ndarray[Any, Any]:
    data = np.asarray(x, dtype=np.float64)
    out = np.empty_like(data)
    for index, value in np.ndenumerate(data):
        shifted = float(value) + bias
        out[index] = 1.0 / (1.0 + np.exp(-shifted))
    return out


def _candidate_affine_sigmoid(
    x: np.ndarray[Any, Any] | list[float], bias: float = 0.0
) -> np.ndarray[Any, Any]:
    data = np.asarray(x, dtype=np.float64)
    shifted = data + bias
    return np.reciprocal(1.0 + np.exp(-shifted))


def make_demo_sigmoid_route() -> AlternativePathRoute[np.ndarray[Any, Any]]:
    """Create a self-contained demo route for benchmarking and comparison."""
    return AlternativePathRoute(
        name="demo.affine-sigmoid",
        baseline=_baseline_affine_sigmoid,
        candidate=_candidate_affine_sigmoid,
        summary="Loop baseline vs vectorised NumPy sigmoid candidate",
        expected_behavior="Candidate should numerically match the baseline and usually run faster",
    )


def build_demo_registry() -> AlternativePathRegistry:
    """Create a registry with the built-in demo route."""
    registry = AlternativePathRegistry()
    registry.register(make_demo_sigmoid_route())
    return registry

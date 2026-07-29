# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — independent retained-project-map oracle

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from sc_neurocore.accel.sc_adaptive_threshold_map import (
    simulate_sc_adaptive_threshold_map,
)

_TRACE = (
    Path(__file__).resolve().parents[1]
    / "src/sc_neurocore/neurons/reference_trace_data/sc_adaptive_threshold_map_spec.json"
)


def _iterate(receipt: dict[str, object]) -> tuple[list[float], list[float], list[int]]:
    """Iterate the written project specification without importing its model."""
    config = dict(receipt["configuration"])
    x = float(config["x"])
    theta = float(config["theta"])
    xs: list[float] = []
    thetas: list[float] = []
    events: list[int] = []
    for current in receipt["current"]:
        sigmoid = 1.0 / (1.0 + math.exp(-4.0 * (x - theta)))
        next_x = min(5.0, max(-5.0, -x + float(config["k"]) * sigmoid + current))
        next_theta = min(
            5.0,
            max(
                -5.0,
                float(config["beta"]) * theta
                + float(config["gamma"]) * int(x >= float(config["theta_spike"])),
            ),
        )
        events.append(int(x < float(config["x_threshold"]) <= next_x))
        x, theta = next_x, next_theta
        xs.append(x)
        thetas.append(theta)
    return xs, thetas, events


def test_independent_project_equation_matches_committed_trace() -> None:
    receipt = json.loads(_TRACE.read_text(encoding="utf-8"))
    xs, thetas, events = _iterate(receipt)
    np.testing.assert_allclose(xs, receipt["x"], rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(thetas, receipt["theta"], rtol=0.0, atol=1e-15)
    assert events == receipt["spikes"]


def test_python_model_matches_project_specification_oracle() -> None:
    receipt = json.loads(_TRACE.read_text(encoding="utf-8"))
    result = simulate_sc_adaptive_threshold_map(
        **receipt["configuration"], current=receipt["current"], backend="python"
    )
    np.testing.assert_allclose(result["x"], receipt["x"], rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(result["theta"], receipt["theta"], rtol=0.0, atol=1e-15)
    np.testing.assert_array_equal(result["spikes"], receipt["spikes"])

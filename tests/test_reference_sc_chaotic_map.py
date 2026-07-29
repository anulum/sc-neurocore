# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — independent SC chaotic-map project-spec oracle

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from sc_neurocore.accel.sc_chaotic_map import simulate_sc_chaotic_map

_TRACE = (
    Path(__file__).resolve().parents[1]
    / "src/sc_neurocore/neurons/reference_trace_data/sc_chaotic_map_spec.json"
)


def _iterate(receipt: dict[str, Any]) -> tuple[list[float], list[float], list[int]]:
    """Iterate the frozen project recurrence without importing its model."""
    config = dict(receipt["configuration"])
    x = float(config["x"])
    y = float(config["y"])
    xs: list[float] = []
    ys: list[float] = []
    events: list[int] = []
    for current in receipt["current"]:
        sigmoid = 1.0 / (1.0 + math.exp(-(x + float(config["alpha"]))))
        next_x = min(
            10.0,
            max(
                -10.0,
                float(config["k_f"]) * x * sigmoid - y + float(current),
            ),
        )
        next_y = min(
            10.0,
            max(-10.0, float(config["k_s"]) * y + float(config["delta"]) * x),
        )
        events.append(int(x < float(config["x_threshold"]) <= next_x))
        x, y = next_x, next_y
        xs.append(x)
        ys.append(y)
    return xs, ys, events


def test_independent_project_equation_matches_committed_trace() -> None:
    receipt = json.loads(_TRACE.read_text(encoding="utf-8"))
    assert receipt["identity"] == "SC-NeuroCore project model; no publication attribution"
    xs, ys, events = _iterate(receipt)
    np.testing.assert_allclose(xs, receipt["x"], rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(ys, receipt["y"], rtol=0.0, atol=1e-15)
    assert events == receipt["spikes"]


def test_python_model_matches_project_specification_oracle() -> None:
    receipt = json.loads(_TRACE.read_text(encoding="utf-8"))
    result = simulate_sc_chaotic_map(
        **receipt["configuration"], current=receipt["current"], backend="python"
    )
    np.testing.assert_allclose(result["x"], receipt["x"], rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(result["y"], receipt["y"], rtol=0.0, atol=1e-15)
    np.testing.assert_array_equal(result["spikes"], receipt["spikes"])

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — primary-equation Aihara oracle

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sc_neurocore.accel.aihara_map import simulate_aihara_map

_TRACE = (
    Path(__file__).resolve().parents[1]
    / "src/sc_neurocore/neurons/reference_trace_data/aihara_map_primary.json"
)


def test_independent_equation_oracle_matches_committed_trace() -> None:
    receipt = json.loads(_TRACE.read_text())
    configuration = receipt["configuration"]
    y = float(configuration["y0"])
    expected_y: list[float] = []
    expected_x: list[float] = []
    expected_events: list[int] = []
    for current in receipt["current"]:
        argument = y / configuration["epsilon"]
        x = (
            1.0 / (1.0 + np.exp(-argument))
            if argument >= 0.0
            else np.exp(argument) / (1.0 + np.exp(argument))
        )
        y = configuration["k"] * y - configuration["alpha"] * x + configuration["bias"] + current
        argument = y / configuration["epsilon"]
        x = (
            1.0 / (1.0 + np.exp(-argument))
            if argument >= 0.0
            else np.exp(argument) / (1.0 + np.exp(argument))
        )
        expected_y.append(y)
        expected_x.append(x)
        expected_events.append(int(x >= 0.5))
    np.testing.assert_allclose(expected_y, receipt["y"], rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(expected_x, receipt["x"], rtol=0.0, atol=1.0e-12)
    assert expected_events == receipt["spikes"]


def test_python_model_matches_primary_equation_oracle() -> None:
    receipt = json.loads(_TRACE.read_text())
    config = receipt["configuration"]
    result = simulate_aihara_map(
        config["y0"],
        config["k"],
        config["alpha"],
        config["bias"],
        config["epsilon"],
        receipt["current"],
        backend="python",
    )
    np.testing.assert_allclose(result["y"], receipt["y"], rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(result["x"], receipt["x"], rtol=0.0, atol=1.0e-15)
    np.testing.assert_array_equal(result["spikes"], receipt["spikes"])

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Strict model-zoo plugin contract tests

"""Strict contract tests for the legacy model-zoo plugin surface."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from sc_neurocore.model_zoo.model_zoo import (
    AdExPlugin,
    HodgkinHuxleyPlugin,
    LIFPlugin,
    NeuronState,
)


@pytest.mark.parametrize(
    ("current_trace", "message"),
    [
        (cast(np.ndarray[Any, Any], [0.0]), "current_trace must be a numpy array"),
        (np.asarray([[0.0, 1.0]], dtype=np.float64), "current_trace must be one-dimensional"),
        (np.asarray([np.nan], dtype=np.float64), "current_trace must contain finite values"),
        (np.asarray([np.inf], dtype=np.float64), "current_trace must contain finite values"),
        (np.asarray([True], dtype=np.bool_), "current_trace must be numeric"),
        (np.asarray(["bad"], dtype=np.str_), "current_trace must be numeric"),
    ],
)
def test_plugin_simulate_rejects_invalid_current_trace(
    current_trace: np.ndarray[Any, Any], message: str
) -> None:
    """Malformed current traces fail before neuron dynamics run."""
    with pytest.raises(ValueError, match=message):
        LIFPlugin().simulate(current_trace)


@pytest.mark.parametrize("dt", [0.0, -0.001, float("nan"), float("inf")])
def test_plugin_simulate_rejects_invalid_timestep(dt: float) -> None:
    """Simulation timesteps must be finite and positive."""
    with pytest.raises(ValueError, match="dt must be finite and positive"):
        LIFPlugin().simulate(np.asarray([0.0], dtype=np.float64), dt=dt)


@pytest.mark.parametrize(
    ("params", "message"),
    [
        (cast(dict[object, object], [("tau_m", 0.02)]), "params must be a mapping"),
        (
            {"tau_m": 0.02, "V_rest": -0.07, "V_thresh": -0.055, "V_reset": -0.075, "R_m": np.nan},
            "params must contain finite real values",
        ),
        (
            {"tau_m": 0.02, "V_rest": -0.07, "V_thresh": -0.055, "V_reset": -0.075, "R_m": True},
            "params must contain finite real values",
        ),
        ({1: 0.02}, "params keys must be strings"),
    ],
)
def test_plugin_simulate_rejects_invalid_params(params: dict[object, object], message: str) -> None:
    """Malformed parameter mappings fail before integration."""
    with pytest.raises(ValueError, match=message):
        LIFPlugin().simulate(
            np.asarray([0.0], dtype=np.float64),
            params=cast(dict[str, float], params),
        )


def test_adex_plugin_dynamics_threshold_and_reset_contract() -> None:
    """AdEx ODE, threshold, and reset paths keep finite state updates."""
    plugin = AdExPlugin()
    params = plugin.default_params()
    state = NeuronState({"V": -52.0, "w": 0.02})

    advanced = plugin.ode_dynamics(state, current=0.4, params=params, dt=0.0001)

    assert advanced is not state
    assert np.isfinite(advanced["V"])
    assert np.isfinite(advanced["w"])
    assert advanced["V"] != state["V"]
    assert advanced["w"] != state["w"]
    assert plugin.threshold_check(NeuronState({"V": params["V_peak"]}), params)
    reset = plugin.reset(NeuronState({"V": 25.0, "w": 0.1}), params)
    assert reset["V"] == params["V_reset"]
    assert reset["w"] == 0.1 + params["b"]


def test_hodgkin_huxley_reset_returns_independent_copy() -> None:
    """Hodgkin-Huxley reset is a no-op copy, not an alias."""
    plugin = HodgkinHuxleyPlugin()
    state = NeuronState({"V": 10.0, "m": 0.5, "h": 0.6, "n": 0.7})

    reset = plugin.reset(state, plugin.default_params())
    reset["V"] = -65.0

    assert state["V"] == 10.0
    assert reset["V"] == -65.0


def test_hodgkin_huxley_threshold_contract() -> None:
    """Hodgkin-Huxley threshold uses the configured voltage cutoff."""
    plugin = HodgkinHuxleyPlugin()
    params = plugin.default_params()

    assert not plugin.threshold_check(NeuronState({"V": params["V_thresh"] - 0.1}), params)
    assert plugin.threshold_check(NeuronState({"V": params["V_thresh"]}), params)

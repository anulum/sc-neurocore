# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — the Studio E-I network is the Brunel-style model its page describes

"""Physics of the Studio's balanced E-I network, on both implementations.

The network had delta-synapse jumps multiplied by the step size and a single
external input per neuron, so no setting reachable from the Studio produced a
spike. These tests pin the model instead of numbers: threshold drive from the
external inputs, silence below it, rates independent of the step size, and a
mean rate that counts silent time. Each runs on the NumPy implementation and,
when the engine is installed, on the Rust one.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio import network
from sc_neurocore.studio.app import create_app

Simulator = Callable[..., dict[str, Any]]


def _numpy(**kwargs: Any) -> dict[str, Any]:
    return network._simulate_numpy(**_arguments(kwargs))


def _rust(**kwargs: Any) -> dict[str, Any]:
    try:
        return network._simulate_rust(**_arguments(kwargs))
    except ImportError:
        pytest.skip("Rust E-I network engine not installed")


def _arguments(overrides: dict[str, Any]) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "n_exc": 80,
        "n_inh": 20,
        "w_ee": 0.1,
        "w_ei": 0.4,
        "w_ie": 0.1,
        "w_ii": 0.4,
        "p_conn": 0.2,
        "ext_rate": network.DEFAULT_EXTERNAL_RATE_HZ,
        "duration": 500.0,
        "dt": 0.1,
    }
    arguments.update(overrides)
    return arguments


BACKENDS = pytest.mark.parametrize("simulate", (_numpy, _rust), ids=("numpy", "rust"))


def test_external_drive_constants_give_a_threshold_inside_the_slider_range() -> None:
    threshold_gap_mv = 15.0  # -50 mV threshold, -65 mV rest
    tau_m_s = 0.020
    threshold_rate = threshold_gap_mv / (
        network.EXTERNAL_WEIGHT_MV * network.EXTERNAL_SYNAPSES * tau_m_s
    )
    assert threshold_rate == pytest.approx(9.375)
    assert threshold_rate < network.DEFAULT_EXTERNAL_RATE_HZ


@BACKENDS
def test_default_network_fires_at_the_rate_the_drive_predicts(simulate: Simulator) -> None:
    # Mean drive 0.1 mV x 800 x 12 Hz x 20 ms = 19.2 mV against a 15 mV gap:
    # a lone neuron fires every 20 ms x ln(19.2 / 4.2) + 2 ms, about 31 Hz, and
    # the default weights (g = 4, four excitatory inputs per inhibitory one)
    # cancel on average.
    result = simulate()
    assert 20.0 < result["mean_exc_rate"] < 45.0
    assert 20.0 < result["mean_inh_rate"] < 45.0


@BACKENDS
def test_drive_well_below_threshold_is_silent(simulate: Simulator) -> None:
    assert simulate(ext_rate=2.0)["n_spikes"] == 0


@BACKENDS
def test_rate_grows_with_the_external_drive(simulate: Simulator) -> None:
    rates = [simulate(ext_rate=rate)["mean_exc_rate"] for rate in (5.0, 10.0, 15.0, 25.0)]
    assert rates == sorted(rates)
    assert rates[-1] > rates[1] > 0.0


@BACKENDS
def test_rate_does_not_depend_on_the_step_size(simulate: Simulator) -> None:
    coarse = simulate(duration=1000.0, dt=0.1)["mean_exc_rate"]
    fine = simulate(duration=1000.0, dt=0.05)["mean_exc_rate"]
    assert fine == pytest.approx(coarse, rel=0.1)


@BACKENDS
def test_mean_rate_counts_silent_time(simulate: Simulator) -> None:
    result = simulate()
    excitatory_spikes = sum(1 for neuron in result["spike_neurons"] if int(neuron) < 80)
    assert result["mean_exc_rate"] == pytest.approx(excitatory_spikes / 80 / 0.5, abs=0.051)


@BACKENDS
def test_stronger_inhibition_lowers_the_excitatory_rate(simulate: Simulator) -> None:
    # w_ei is inhibitory-to-excitatory in the model's post-pre convention.
    weak = simulate(w_ei=0.1)["mean_exc_rate"]
    strong = simulate(w_ei=1.0)["mean_exc_rate"]
    assert strong < weak


@BACKENDS
def test_a_very_large_external_mean_terminates(simulate: Simulator) -> None:
    # 800 inputs x 50 Hz x 5 ms = 200 expected events per step.
    assert simulate(ext_rate=50.0, duration=50.0, dt=5.0)["n_spikes"] > 0


def test_the_route_refuses_a_negative_external_rate() -> None:
    client = TestClient(create_app(), base_url="http://127.0.0.1")
    response = client.post("/api/network/ei", json={"ext_rate": -1.0})
    assert response.status_code == 422

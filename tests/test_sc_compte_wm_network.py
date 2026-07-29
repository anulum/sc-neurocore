# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — executable 2,560-cell SC Compte network tests

"""Focused full-population dynamics, parity, protocol, and receipt tests."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.network import (
    SCCompteWMNetwork,
    SCCompteWMNetworkSpec,
    SCCompteWMStimulus,
)
from sc_neurocore.neurons.models.compte_wm import CompteWMNeuron


def _zero_events() -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    return np.zeros(2048, dtype=np.int64), np.zeros(512, dtype=np.int64)


def test_network_enrols_every_cell_and_returns_defensive_state() -> None:
    network = SCCompteWMNetwork()
    state = network.state()
    assert state.v_exc_mv.shape == (2048,)
    assert state.v_inh_mv.shape == (512,)
    assert state.recurrent_nmda.shape == (2048,)
    assert state.recurrent_gabaa.shape == (512,)
    state.v_exc_mv[0] = 0.0
    assert network.state().v_exc_mv[0] == -70.0
    assert network.spec.allow_recurrent_autapses is False


def test_isolated_excitatory_external_impulse_matches_preserved_cell_model() -> None:
    network = SCCompteWMNetwork()
    exc_events, inh_events = _zero_events()
    exc_events[17] = 1
    receipt = network.step(
        external_exc_events=exc_events,
        external_inh_events=inh_events,
    )
    original = CompteWMNeuron()
    assert original.step(external_spike=True) == 0
    state = network.state()
    assert state.v_exc_mv[17] == pytest.approx(original.v, rel=0.0, abs=2e-14)
    assert state.external_ampa_exc[17] == pytest.approx(original.s_ampa, abs=2e-14)
    assert receipt.excitatory_input.total_events == 1
    assert not np.any(receipt.excitatory_spikes)


def test_fft_ring_coupling_matches_independent_dense_target_oracle() -> None:
    spec = SCCompteWMNetworkSpec()
    seed_network = SCCompteWMNetwork(spec)
    state = seed_network.state()
    source = np.zeros(2048, dtype=np.float64)
    source[[0, 37, 1024, 1901]] = [0.2, 0.4, 0.1, 0.3]
    state.recurrent_nmda[:] = source
    state.v_exc_mv[:] = -60.0
    network = SCCompteWMNetwork(spec, state=state)
    exc_events, inh_events = _zero_events()
    network.step(external_exc_events=exc_events, external_inh_events=inh_events)

    dt = spec.dt_ms
    target = 113
    kernel = spec.connectivity_footprint("ee", 0.0, spec.preferred_angles_deg("excitatory"))
    indices = (target - np.arange(2048)) % 2048
    aggregate_0 = float(np.dot(kernel[indices], source) - kernel[0] * source[target])
    source_mid = source * (1.0 - 0.5 * dt / spec.tau_nmda_ms)
    aggregate_mid = float(np.dot(kernel[indices], source_mid) - kernel[0] * source_mid[target])
    v0 = -60.0
    g_l = spec.excitatory.leak_conductance_ns / 1000.0
    g_nmda = spec.recurrent_conductance_ns("ee") / 1000.0

    def dv(voltage: float, aggregate: float) -> float:
        block = 1.0 / (1.0 + spec.magnesium_mm / 3.57 * math.exp(-0.062 * voltage))
        return (
            -g_l * (voltage - spec.excitatory.leak_reversal_mv)
            - g_nmda * aggregate * block * voltage
        ) / spec.excitatory.capacitance_nf

    midpoint_v = v0 + 0.5 * dt * dv(v0, aggregate_0)
    expected = v0 + dt * dv(midpoint_v, aggregate_mid)
    assert network.state().v_exc_mv[target] == pytest.approx(expected, rel=0.0, abs=2e-13)


def test_counter_driven_runs_are_repeatable_and_seed_separated() -> None:
    first = SCCompteWMNetwork().run(0.1, statistics_window_ms=0.1)
    second = SCCompteWMNetwork().run(0.1, statistics_window_ms=0.1)
    third = SCCompteWMNetwork(SCCompteWMNetworkSpec(seed=43)).run(0.1, statistics_window_ms=0.1)
    assert first == second
    assert first.input_sha256 != third.input_sha256
    assert first.final_state_sha256 != third.final_state_sha256


def test_global_protocol_current_spikes_and_refractory_clamps_full_population() -> None:
    network = SCCompteWMNetwork()
    stimulus = SCCompteWMStimulus(
        start_ms=0.0,
        duration_ms=0.02,
        current_pa=600_000.0,
        kind="global_current",
        center_deg=None,
    )
    receipt = network.run(0.02, stimuli=(stimulus,), statistics_window_ms=0.02)
    assert receipt.excitatory_spikes == 2048
    assert receipt.windows[0].statistics is not None
    state = network.state()
    assert np.all(state.v_exc_mv == network.spec.excitatory.reset_mv)
    assert np.all(state.refractory_exc_ms == network.spec.excitatory.refractory_ms)
    exc_events, inh_events = _zero_events()
    network.step(external_exc_events=exc_events, external_inh_events=inh_events)
    assert np.all(network.state().v_exc_mv == network.spec.excitatory.reset_mv)


def test_step_validation_is_atomic_and_requires_complete_event_override() -> None:
    network = SCCompteWMNetwork()
    before = network.state().sha256()
    with pytest.raises(ValueError, match="supplied together"):
        network.step(external_exc_events=np.zeros(2048, dtype=np.int64))
    assert network.state().sha256() == before
    current = np.zeros(2048, dtype=np.float64)
    current[4] = np.nan
    with pytest.raises(ValueError, match="finite"):
        network.step(current)
    assert network.state().sha256() == before


def test_protocol_epochs_fail_closed_outside_run() -> None:
    stimulus = SCCompteWMStimulus(0.08, 0.04, 200.0, center_deg=90.0)
    with pytest.raises(ValueError, match="within"):
        SCCompteWMNetwork().run(0.1, stimuli=(stimulus,), statistics_window_ms=0.1)

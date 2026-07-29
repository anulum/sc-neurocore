# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC Compte network specification contracts

"""Focused identity, topology, protocol, and statistics tests for the SC mod."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from sc_neurocore.network import (
    SCCompteProtocolSpec,
    SCCompteWMNetworkSpec,
    circular_displacement_deg,
    circular_distance_deg,
    summarize_activity,
)


def test_identity_and_population_are_fixed() -> None:
    spec = SCCompteWMNetworkSpec()
    assert spec.identity == "SC-COMPTE-WM-NETWORK"
    assert spec.n_excitatory == 2048
    assert spec.n_inhibitory == 512
    assert spec.n_cells == 2560
    with pytest.raises(ValueError, match="fixed"):
        replace(spec, n_excitatory=2047)


def test_preferred_cues_uniformly_cover_each_ring() -> None:
    spec = SCCompteWMNetworkSpec()
    exc = spec.preferred_angles_deg("excitatory")
    inh = spec.preferred_angles_deg("inhibitory")
    assert exc.shape == (2048,)
    assert inh.shape == (512,)
    assert exc[0] == 0.0
    assert exc[-1] == pytest.approx(360.0 - 360.0 / 2048)
    assert np.diff(exc) == pytest.approx(np.full(2047, 360.0 / 2048))


def test_shortest_circular_distance_and_displacement_wrap() -> None:
    distance = circular_distance_deg([359.0, 1.0, 180.0], 0.0)
    assert distance.tolist() == pytest.approx([-1.0, 1.0, -180.0])
    assert circular_displacement_deg(350.0, 10.0) == 20.0
    assert circular_displacement_deg(10.0, 350.0) == -20.0


def test_ee_footprint_has_exact_unit_mean_and_local_peak() -> None:
    spec = SCCompteWMNetworkSpec()
    targets = spec.preferred_angles_deg("excitatory")
    weights = spec.connectivity_footprint("ee", 0.0, targets)
    assert float(np.mean(weights)) == pytest.approx(1.0, abs=2e-15)
    assert weights[0] == pytest.approx(spec.ee_j_plus, rel=2e-12)
    assert np.all(weights > 0.0)
    assert weights[0] > weights[len(weights) // 2]


def test_only_selected_control_projection_is_structured() -> None:
    spec = SCCompteWMNetworkSpec()
    targets = spec.preferred_angles_deg("inhibitory")
    assert np.array_equal(spec.connectivity_footprint("ei", 0.0, targets), np.ones(512))
    tuned = replace(spec, structured_ei=True)
    weights = tuned.connectivity_footprint("ei", 0.0, targets)
    assert float(np.mean(weights)) == pytest.approx(1.0, abs=2e-15)
    assert weights[0] == pytest.approx(spec.ei_j_plus, rel=2e-12)
    assert np.array_equal(spec.connectivity_footprint("ie", 0.0, targets), np.ones(512))


def test_modulated_set_scales_nmda_and_gabaa_separately() -> None:
    control = SCCompteWMNetworkSpec()
    modulated = replace(control, modulated=True)
    assert modulated.recurrent_conductance_ns("ee") == pytest.approx(0.381 * 1.2)
    assert modulated.recurrent_conductance_ns("ei") == pytest.approx(0.292 * 1.2)
    assert modulated.recurrent_conductance_ns("ie") == pytest.approx(1.336 * 1.4)
    assert modulated.recurrent_conductance_ns("ii") == pytest.approx(1.024 * 1.4)


def test_sc_cue_profile_is_periodic_compact_and_peak_bound() -> None:
    spec = SCCompteWMNetworkSpec()
    current = spec.cue_current_pa(359.0, [359.0, 0.0, 17.0, 18.0, 180.0])
    assert current[0] == pytest.approx(200.0)
    assert 0.0 < current[1] < 200.0
    assert current[2] == 0.0
    assert current[3] == 0.0
    assert current[4] == 0.0


def test_activity_summary_recovers_ring_angle_rates_and_width() -> None:
    spec = SCCompteWMNetworkSpec()
    exc = np.zeros(spec.n_excitatory, dtype=np.int64)
    inh = np.ones(spec.n_inhibitory, dtype=np.int64)
    exc[512] = 20  # 90 degrees
    stats = summarize_activity(spec, exc, inh, window_ms=500.0)
    assert stats.bump_angle_deg == pytest.approx(90.0)
    assert stats.resultant_length == pytest.approx(1.0)
    assert stats.circular_width_deg == pytest.approx(0.0)
    assert stats.excitatory_rate_hz == pytest.approx(20 / (2048 * 0.5))
    assert stats.inhibitory_rate_hz == pytest.approx(2.0)


def test_statistics_fail_closed_on_empty_or_wrong_shape() -> None:
    spec = SCCompteWMNetworkSpec()
    with pytest.raises(ValueError, match="at least one"):
        summarize_activity(spec, np.zeros(2048), np.zeros(512), 500.0)
    with pytest.raises(ValueError, match="match"):
        summarize_activity(spec, np.ones(8), np.ones(512), 500.0)


def test_protocol_rejects_non_physical_widths() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        SCCompteProtocolSpec(cue_half_width_deg=181.0)

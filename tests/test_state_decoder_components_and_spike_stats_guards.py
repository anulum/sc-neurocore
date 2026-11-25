# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — StateDecoder connected-component and spike-train statistic guard tests

"""Verify transitive attractor components in StateDecoder and spike-statistic guards."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from sc_neurocore.identity.decoder import StateDecoder
from sc_neurocore.identity.substrate import IdentitySubstrate
from sc_neurocore.spike_train_statistics import (
    bin_spike_train,
    cv_isi,
    fano_factor,
    firing_rate,
    isi,
    permutation_entropy,
    power_spectrum,
    spike_count,
    spike_times,
)


def _drive_chain_substrate(channel_order: tuple[int, ...]) -> IdentitySubstrate:
    """Drive an 8-neuron cortical substrate with a 3-node correlation chain."""
    substrate = IdentitySubstrate(n_cortical=8, n_inhibitory=4, n_memory=4, seed=11)
    n_steps = 140
    base_channels = np.zeros((8, n_steps), dtype=np.float64)
    base_channels[0, 0:40] = 18.0
    base_channels[1, 90:130] = 18.0
    base_channels[2, 0:40] = 18.0
    base_channels[2, 90:130] = 18.0

    for t in range(n_steps):
        currents = np.array([base_channels[ch, t] for ch in channel_order], dtype=np.float64)
        substrate.step(currents)
    return substrate


def test_extract_attractor_states_connected_components_are_disjoint_and_invariant() -> None:
    """Connected-component clustering merges transitive chains regardless of neuron order."""
    sub_bridge_last = _drive_chain_substrate((0, 1, 2, 3, 4, 5, 6, 7))
    sub_bridge_first = _drive_chain_substrate((2, 0, 1, 3, 4, 5, 6, 7))

    fc = StateDecoder(sub_bridge_last).extract_connectivity_signature()
    assert fc[0, 2] > 0.4
    assert fc[1, 2] > 0.4
    assert fc[0, 1] < 0.2

    comp_last = StateDecoder(sub_bridge_last).extract_attractor_states(threshold=0.4)
    comp_first = StateDecoder(sub_bridge_first).extract_attractor_states(threshold=0.4)

    assert len(comp_last) == 1
    assert len(comp_first) == 1
    np.testing.assert_array_equal(comp_last[0], np.array([0, 1, 2], dtype=np.int64))
    np.testing.assert_array_equal(comp_first[0], np.array([0, 1, 2], dtype=np.int64))


def test_extract_attractor_states_stochastic_drive_produces_disjoint_sorted_groups() -> None:
    """Every neuron belongs to at most one sorted attractor component."""
    substrate = IdentitySubstrate(n_cortical=16, n_inhibitory=4, n_memory=4, seed=1)
    rng = np.random.default_rng(1)
    for _ in range(80):
        substrate.step((rng.random(16) < 0.25).astype(np.float64) * 18.0)

    decoder = StateDecoder(substrate)
    attractors = decoder.extract_attractor_states(threshold=0.22)
    assert len(attractors) >= 1

    seen: list[int] = []
    mins: list[int] = []
    for group in attractors:
        assert group.dtype == np.int64
        assert group.size >= 2
        indices = [int(x) for x in group]
        assert indices == sorted(indices)
        mins.append(indices[0])
        seen.extend(indices)

    assert mins == sorted(mins)
    assert len(seen) == len(set(seen))


@pytest.mark.parametrize("bad_threshold", [float("nan"), float("inf"), float("-inf"), True, False])
def test_extract_attractor_states_rejects_nonfinite_or_boolean_threshold(
    bad_threshold: Any,
) -> None:
    """Non-finite or boolean correlation thresholds raise ValueError."""
    substrate = IdentitySubstrate(n_cortical=8, n_inhibitory=4, n_memory=4, seed=3)
    decoder = StateDecoder(substrate)
    with pytest.raises(ValueError, match="threshold"):
        decoder.extract_attractor_states(threshold=bad_threshold)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_spike_train_statistics_reject_nonfinite_samples(bad_value: float) -> None:
    """All nine spike-train statistics reject NaN and infinite train entries."""
    train = np.array([0.0, 1.0, bad_value, 0.0, 1.0, 0.0], dtype=np.float64)
    with pytest.raises(ValueError, match="finite"):
        spike_times(train)
    with pytest.raises(ValueError, match="finite"):
        isi(train)
    with pytest.raises(ValueError, match="finite"):
        firing_rate(train)
    with pytest.raises(ValueError, match="finite"):
        spike_count(train)
    with pytest.raises(ValueError, match="finite"):
        bin_spike_train(train, bin_size=2)
    with pytest.raises(ValueError, match="finite"):
        power_spectrum(train)
    with pytest.raises(ValueError, match="finite"):
        cv_isi(train)
    with pytest.raises(ValueError, match="finite"):
        fano_factor(train, window_ms=2.0)
    with pytest.raises(ValueError, match="finite"):
        permutation_entropy(train, order=2, delay=1)


def test_spike_train_statistics_reject_complex_or_object_samples() -> None:
    """Complex or non-numeric object arrays are rejected with ValueError."""
    complex_train = np.array([1.0 + 0.0j, 0.0 + 1.0j])
    object_train = np.array([1, "bad"], dtype=object)
    with pytest.raises(ValueError, match="finite"):
        spike_count(complex_train)
    with pytest.raises(ValueError, match="finite"):
        firing_rate(object_train)


@pytest.mark.parametrize("bad_dt", [-0.001, float("nan"), float("inf"), float("-inf"), True, False])
def test_non_negative_dt_functions_reject_negative_nonfinite_or_bool_dt(bad_dt: Any) -> None:
    """Functions allowing dt == 0 reject negative, non-finite, or boolean dt."""
    train = np.array([1, 0, 1, 0, 1], dtype=np.int8)
    with pytest.raises(ValueError, match="dt"):
        spike_times(train, dt=bad_dt)
    with pytest.raises(ValueError, match="dt"):
        isi(train, dt=bad_dt)
    with pytest.raises(ValueError, match="dt"):
        firing_rate(train, dt=bad_dt)
    with pytest.raises(ValueError, match="dt"):
        cv_isi(train, dt=bad_dt)


def test_zero_dt_contracts_preserved_for_rate_times_isi_and_cv() -> None:
    """Zero dt and numpy real/integer scalars remain accepted across statistics and decoder."""
    train = np.array([1, 0, 1, 0, 1], dtype=np.int8)
    np.testing.assert_array_equal(spike_times(train, dt=0.0), np.zeros(3, dtype=np.float64))
    np.testing.assert_array_equal(isi(train, dt=0.0), np.zeros(2, dtype=np.float64))
    assert firing_rate(train, dt=0.0) == 0.0
    assert math.isnan(cv_isi(train, dt=0.0))

    dt_f32: Any = np.float32(0.001)
    dt_i64: Any = np.int64(1)
    win_f32: Any = np.float32(2.0)
    assert spike_times(train, dt=dt_f32).size == 3
    assert isi(train, dt=dt_i64).size == 2
    assert firing_rate(train, dt=dt_f32) > 0.0
    assert cv_isi(train, dt=dt_f32) >= 0.0
    psd, freqs = power_spectrum(train, dt=dt_f32)
    assert psd.size == freqs.size > 0
    assert fano_factor(train, window_ms=win_f32, dt=dt_f32) >= 0.0

    sub = IdentitySubstrate(n_cortical=8, n_inhibitory=4, n_memory=4, seed=5)
    for _ in range(60):
        sub.step(np.full(8, 18.0, dtype=np.float64))
    thr_f32: Any = np.float32(0.5)
    thr_i64: Any = np.int64(0)
    assert isinstance(StateDecoder(sub).extract_attractor_states(threshold=thr_f32), list)
    assert isinstance(StateDecoder(sub).extract_attractor_states(threshold=thr_i64), list)
    for bad_complex in (1.0 + 0.0j, np.complex128(0.5)):
        bad_c: Any = bad_complex
        with pytest.raises(ValueError, match="threshold"):
            StateDecoder(sub).extract_attractor_states(threshold=bad_c)
        with pytest.raises(ValueError, match="dt"):
            firing_rate(train, dt=bad_c)
        with pytest.raises(ValueError, match="dt"):
            power_spectrum(train, dt=bad_c)


@pytest.mark.parametrize(
    "bad_dt", [0.0, -0.001, float("nan"), float("inf"), float("-inf"), True, False]
)
def test_strictly_positive_dt_required_for_power_spectrum_and_fano_factor(bad_dt: Any) -> None:
    """power_spectrum and fano_factor reject zero, negative, non-finite, or boolean dt."""
    train = np.array([1, 0, 1, 0, 1, 0], dtype=np.int8)
    with pytest.raises(ValueError, match="dt"):
        power_spectrum(train, dt=bad_dt)
    with pytest.raises(ValueError, match="dt"):
        fano_factor(train, window_ms=2.0, dt=bad_dt)


@pytest.mark.parametrize(
    "bad_window", [0.0, -5.0, float("nan"), float("inf"), float("-inf"), True, False]
)
def test_fano_factor_rejects_nonpositive_nonfinite_or_bool_window(bad_window: Any) -> None:
    """fano_factor requires a finite positive window_ms."""
    train = np.array([1, 0, 1, 0, 1, 0], dtype=np.int8)
    with pytest.raises(ValueError, match="window_ms"):
        fano_factor(train, window_ms=bad_window, dt=0.001)


@pytest.mark.parametrize("bad_bin", [0, -4, True, False, 2.5])
def test_bin_spike_train_rejects_nonpositive_bool_or_fractional_bin_size(bad_bin: Any) -> None:
    """bin_spike_train requires a positive integer bin_size."""
    train = np.array([1, 0, 1, 1, 0, 1], dtype=np.int8)
    with pytest.raises(ValueError, match="bin_size"):
        bin_spike_train(train, bin_size=bad_bin)


@pytest.mark.parametrize(
    ("bad_order", "bad_delay"),
    [(0, 1), (3, 0), (-1, 1), (3, -2), (True, 1), (3, False), (2.5, 1), (3, 1.5)],
)
def test_permutation_entropy_rejects_invalid_or_boolean_order_and_delay(
    bad_order: Any, bad_delay: Any
) -> None:
    """permutation_entropy requires positive non-boolean integers for order and delay."""
    train = np.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=np.int8)
    with pytest.raises(ValueError, match="order|delay"):
        permutation_entropy(train, order=bad_order, delay=bad_delay)

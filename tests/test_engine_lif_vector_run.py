# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Rust engine vector LIF run contracts

"""Rust engine vector LIF run contracts.

These tests exercise the public engine operations that run fixed-point LIF
dynamics over a time vector. They assert parity with per-step execution,
constant-current equivalence, empty-input boundaries, and non-degenerate
membrane evolution.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sc_neurocore_engine", reason="Rust engine not built", exc_type=ImportError)

import sc_neurocore_engine as v3
from sc_neurocore_engine import FixedPointLif


def test_vector_lif_run_matches_per_step_fixed_point_lif() -> None:
    n_steps = 200
    leak_k = 20
    gain_k = 256
    input_current = 128

    lif = FixedPointLif()
    per_step_spikes = []
    per_step_voltages = []
    for _ in range(n_steps):
        spike, voltage = lif.step(leak_k, gain_k, input_current, 0)
        per_step_spikes.append(spike)
        per_step_voltages.append(voltage)

    spikes, voltages = v3.batch_lif_run(
        n_steps,
        leak_k=leak_k,
        gain_k=gain_k,
        i_t=input_current,
    )

    assert spikes.dtype == np.int32
    assert voltages.dtype == np.int16
    np.testing.assert_array_equal(spikes, per_step_spikes)
    np.testing.assert_array_equal(voltages, per_step_voltages)


def test_vector_lif_run_with_current_array_matches_constant_current_operation() -> None:
    n_steps = 100
    currents = np.full(n_steps, 128, dtype=np.int16)

    constant_spikes, constant_voltages = v3.batch_lif_run(
        n_steps,
        leak_k=20,
        gain_k=256,
        i_t=128,
    )
    varying_spikes, varying_voltages = v3.batch_lif_run_varying(
        leak_k=20,
        gain_k=256,
        currents=currents,
    )

    np.testing.assert_array_equal(varying_spikes, constant_spikes)
    np.testing.assert_array_equal(varying_voltages, constant_voltages)


def test_vector_lif_run_accepts_zero_steps_without_padding() -> None:
    spikes, voltages = v3.batch_lif_run(0, leak_k=20, gain_k=256, i_t=128)

    assert spikes.shape == (0,)
    assert voltages.shape == (0,)


def test_vector_lif_run_strong_current_changes_membrane_state() -> None:
    spikes, voltages = v3.batch_lif_run(100, leak_k=20, gain_k=256, i_t=200)

    assert spikes.shape == (100,)
    assert voltages.shape == (100,)
    assert len(np.unique(voltages)) > 1


def test_vector_lif_run_honours_fixed_point_configuration() -> None:
    spikes, voltages = v3.batch_lif_run(
        50,
        leak_k=10,
        gain_k=512,
        i_t=100,
        data_width=16,
        fraction=8,
        v_rest=0,
        v_reset=0,
        v_threshold=256,
        refractory_period=3,
    )

    assert spikes.shape == (50,)
    assert voltages.shape == (50,)


def test_vector_lif_run_with_noise_array_preserves_time_axis() -> None:
    currents = np.full(50, 200, dtype=np.int16)
    noises = np.zeros(50, dtype=np.int16)

    spikes, voltages = v3.batch_lif_run_varying(
        leak_k=20,
        gain_k=256,
        currents=currents,
        noises=noises,
    )

    assert spikes.shape == currents.shape
    assert voltages.shape == currents.shape

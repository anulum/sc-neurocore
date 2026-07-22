# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — fixed-point LIF engine-binding contracts

"""Installed-extension contracts for fixed-point LIF batch kernels."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

import sc_neurocore_engine as engine

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def test_exported_names_signatures_and_top_level_identities_are_stable() -> None:
    signatures = {
        "batch_lif_run": (
            "(n_steps, leak_k, gain_k, i_t, noise_in=0, data_width=16, "
            "fraction=8, v_rest=0, v_reset=0, v_threshold=256, refractory_period=2)"
        ),
        "batch_lif_run_multi": (
            "(n_neurons, n_steps, leak_k, gain_k, currents, data_width=16, "
            "fraction=8, v_rest=0, v_reset=0, v_threshold=256, refractory_period=2)"
        ),
        "batch_lif_run_varying": (
            "(leak_k, gain_k, currents, noises=None, data_width=16, fraction=8, "
            "v_rest=0, v_reset=0, v_threshold=256, refractory_period=2)"
        ),
    }

    for name, signature in signatures.items():
        function = getattr(extension, name)
        assert function.__name__ == name
        assert function.__module__ == "sc_neurocore_engine.sc_neurocore_engine"
        assert function.__text_signature__ == signature
        assert getattr(engine, name) is function


def test_constant_batch_matches_stepwise_fixed_point_lif() -> None:
    reference = extension.FixedPointLif()
    expected = [reference.step(20, 256, 128, 0) for _ in range(32)]

    spikes, voltages = extension.batch_lif_run(32, 20, 256, 128)

    np.testing.assert_array_equal(spikes, [result[0] for result in expected])
    np.testing.assert_array_equal(voltages, [result[1] for result in expected])
    assert spikes.dtype == np.int32
    assert voltages.dtype == np.int16


def test_multi_batch_rejects_current_count_mismatch() -> None:
    currents = np.asarray([64, 128], dtype=np.int16)

    with pytest.raises(ValueError, match="does not match n_neurons"):
        extension.batch_lif_run_multi(3, 8, 20, 256, currents)


def test_varying_batch_rejects_noise_count_mismatch() -> None:
    currents = np.asarray([64, 128], dtype=np.int16)
    noises = np.asarray([0], dtype=np.int16)

    with pytest.raises(ValueError, match="does not match currents length"):
        extension.batch_lif_run_varying(20, 256, currents, noises)

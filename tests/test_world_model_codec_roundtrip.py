# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCodecRoundtrip from former test_world_model.py

"""Focused suite: TestCodecRoundtrip from former test_world_model.py."""

from __future__ import annotations

from tests.world_model_support import *  # noqa: F403


class TestCodecRoundtrip:
    def test_lossless_roundtrip(self):
        n_ch = 4
        T = 20
        rng = np.random.RandomState(42)
        spikes = (rng.random((T, n_ch)) < 0.3).astype(np.int8)

        errors, correct = predict_and_xor_world_model(spikes, n_channels=n_ch, seed=0)
        recovered = xor_and_recover_world_model(errors, n_channels=n_ch, seed=0)
        np.testing.assert_array_equal(spikes, recovered)

    def test_correct_count_sane(self):
        n_ch = 4
        T = 30
        rng = np.random.RandomState(0)
        spikes = (rng.random((T, n_ch)) < 0.3).astype(np.int8)
        _, correct = predict_and_xor_world_model(spikes, n_channels=n_ch)
        assert 0 <= correct <= T * n_ch

    def test_errors_are_binary(self):
        n_ch = 2
        T = 20
        spikes = np.zeros((T, n_ch), dtype=np.int8)
        spikes[::2, 0] = 1
        errors, _ = predict_and_xor_world_model(spikes, n_channels=n_ch)
        assert set(np.unique(errors)).issubset({0, 1})

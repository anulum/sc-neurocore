# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRateEncode from former test_encoding.py

"""Focused suite: TestRateEncode from former test_encoding.py."""

from __future__ import annotations

from tests.test_training.encoding_support import *  # noqa: F403


class TestRateEncode:
    def test_output_shape_is_time_major(self):
        values = torch.rand(16, 10)

        spikes = rate_encode(values, n_timesteps=25)

        assert spikes.shape == (25, 16, 10)

    def test_output_is_binary(self):
        spikes = rate_encode(torch.rand(8), n_timesteps=100)

        assert set(spikes.unique().tolist()).issubset({0.0, 1.0})

    def test_higher_probability_produces_more_spikes(self):
        high = rate_encode(torch.tensor([0.9]), n_timesteps=1000)
        low = rate_encode(torch.tensor([0.1]), n_timesteps=1000)

        assert high.sum() > low.sum()

    def test_values_outside_probability_range_are_clamped(self):
        spikes = rate_encode(torch.tensor([1.5, -0.5]), n_timesteps=10)

        assert spikes.shape == (10, 2)
        assert set(spikes.unique().tolist()).issubset({0.0, 1.0})

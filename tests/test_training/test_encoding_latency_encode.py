# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLatencyEncode from former test_encoding.py

"""Focused suite: TestLatencyEncode from former test_encoding.py."""

from __future__ import annotations

from tests.test_training.encoding_support import *  # noqa: F403

class TestLatencyEncode:
    def test_output_shape_is_time_major(self):
        values = torch.rand(8, 4)

        spikes = latency_encode(values, n_timesteps=20)

        assert spikes.shape == (20, 8, 4)

    def test_each_scalar_emits_one_spike(self):
        spikes = latency_encode(torch.tensor([0.5]), n_timesteps=20, tau=5.0)

        assert spikes.sum().item() == 1.0

    def test_larger_value_spikes_no_later_than_smaller_value(self):
        high = latency_encode(torch.tensor([0.95]), n_timesteps=20, tau=5.0)
        low = latency_encode(torch.tensor([0.1]), n_timesteps=20, tau=5.0)

        assert high.squeeze().argmax().item() <= low.squeeze().argmax().item()

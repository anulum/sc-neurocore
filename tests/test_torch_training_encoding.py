# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEncoding from former test_torch_training.py

"""Focused suite: TestEncoding from former test_torch_training.py."""

from __future__ import annotations

from tests.torch_training_support import *  # noqa: F403


class TestEncoding:
    def test_rate_encode_shape(self):
        x = torch.rand(8, 784)
        spikes = rate_encode(x, n_timesteps=25)
        assert spikes.shape == (25, 8, 784)

    def test_rate_encode_binary(self):
        x = torch.rand(4, 16)
        spikes = rate_encode(x, n_timesteps=10)
        assert set(spikes.unique().tolist()).issubset({0.0, 1.0})

    def test_rate_encode_rate_proportional(self):
        """Higher input values should produce more spikes on average."""
        torch.manual_seed(42)
        low = torch.tensor([0.1])
        high = torch.tensor([0.9])
        low_spikes = rate_encode(low, n_timesteps=1000).sum()
        high_spikes = rate_encode(high, n_timesteps=1000).sum()
        assert high_spikes > low_spikes

    def test_latency_encode_shape(self):
        x = torch.rand(8, 16)
        spikes = latency_encode(x, n_timesteps=20)
        assert spikes.shape == (20, 8, 16)

    def test_latency_encode_one_spike(self):
        """Each input neuron should spike exactly once."""
        x = torch.rand(4, 8)
        spikes = latency_encode(x, n_timesteps=20)
        assert (spikes.sum(dim=0) == 1.0).all()

    def test_latency_strong_input_spikes_earlier(self):
        x = torch.tensor([0.9, 0.1])
        spikes = latency_encode(x, n_timesteps=20, tau=5.0)
        first_spike_0 = spikes[:, 0].argmax().item()
        first_spike_1 = spikes[:, 1].argmax().item()
        assert first_spike_0 < first_spike_1

    def test_delta_encode(self):
        x = torch.tensor([[0.0], [0.0], [1.0], [1.0], [0.0]])  # step up, then down
        spikes = delta_encode(x, threshold=0.5)
        assert spikes[2, 0].item() == 1.0  # step up
        assert spikes[4, 0].item() == 1.0  # step down

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEncoding from former test_torch_training.py

"""Focused suite: TestEncoding from former test_torch_training.py."""

from __future__ import annotations

from typing import Any, cast

from tests.torch_training_support import (
    delta_encode,
    latency_encode,
    rate_encode,
    torch,
)


class TestEncoding:
    """Rate, latency and delta spike encoders."""

    def test_rate_encode_shape(self) -> None:
        """Rate encoding of a batch over 25 steps gives a 25x8x784 tensor."""
        x = torch.rand(8, 784)
        spikes = rate_encode(x, n_timesteps=25)
        assert spikes.shape == (25, 8, 784)

    def test_rate_encode_binary(self) -> None:
        """Rate encoding gives only zeros and ones."""
        x = torch.rand(4, 16)
        spikes = rate_encode(x, n_timesteps=10)
        assert set(cast(Any, spikes).unique().tolist()).issubset({0.0, 1.0})

    def test_rate_encode_rate_proportional(self) -> None:
        """Higher input values should produce more spikes on average."""
        torch.manual_seed(42)
        low = torch.tensor([0.1])
        high = torch.tensor([0.9])
        low_spikes = rate_encode(low, n_timesteps=1000).sum()
        high_spikes = rate_encode(high, n_timesteps=1000).sum()
        assert high_spikes > low_spikes

    def test_latency_encode_shape(self) -> None:
        """Latency encoding over 20 steps gives a 20x8x16 tensor."""
        x = torch.rand(8, 16)
        spikes = latency_encode(x, n_timesteps=20)
        assert spikes.shape == (20, 8, 16)

    def test_latency_encode_one_spike(self) -> None:
        """Each input neuron should spike exactly once."""
        x = torch.rand(4, 8)
        spikes = latency_encode(x, n_timesteps=20)
        assert (spikes.sum(dim=0) == 1.0).all()

    def test_latency_strong_input_spikes_earlier(self) -> None:
        """A stronger input spikes earlier under latency encoding."""
        x = torch.tensor([0.9, 0.1])
        spikes = latency_encode(x, n_timesteps=20, tau=5.0)
        first_spike_0 = spikes[:, 0].argmax().item()
        first_spike_1 = spikes[:, 1].argmax().item()
        assert first_spike_0 < first_spike_1

    def test_delta_encode(self) -> None:
        """Delta encoding spikes on the step up and on the step down."""
        x = torch.tensor([[0.0], [0.0], [1.0], [1.0], [0.0]])  # step up, then down
        spikes = delta_encode(x, threshold=0.5)
        assert spikes[2, 0].item() == 1.0  # step up
        assert spikes[4, 0].item() == 1.0  # step down

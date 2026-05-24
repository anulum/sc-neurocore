# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for spike encoding utilities

"""Behavioural contracts for training spike encoders."""

import pytest

torch = pytest.importorskip("torch")

from sc_neurocore.training.encoding import delta_encode, latency_encode, rate_encode


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


class TestDeltaEncode:
    def test_output_shape_matches_input_trace(self):
        values = torch.randn(10, 4)

        spikes = delta_encode(values, threshold=0.1)

        assert spikes.shape == (10, 4)

    def test_constant_signal_is_silent(self):
        values = torch.ones(10, 4) * 5.0

        spikes = delta_encode(values, threshold=0.1)

        assert spikes.sum().item() == 0.0

    def test_step_change_emits_at_transition(self):
        values = torch.zeros(10, 1)
        values[5:] = 1.0

        spikes = delta_encode(values, threshold=0.5)

        assert spikes[5, 0].item() == 1.0

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLatencyEncodeStrict from former test_datasets.py

"""Focused suite: TestLatencyEncodeStrict from former test_datasets.py."""

from __future__ import annotations

from tests.datasets_support import *  # noqa: F403

class TestLatencyEncodeStrict:
    """Input range guard added by task #27."""

    def test_strict_default_raises_on_value_above_one(self):
        import pytest

        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            latency_encode(np.array([0.5, 1.5]), T=50)

    def test_strict_default_raises_on_negative_value(self):
        import pytest

        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            latency_encode(np.array([-0.1, 0.5]), T=50)

    def test_strict_false_keeps_legacy_silent_clip(self):
        # Above-1 values fold to spike-time 0 (clip on the resulting
        # spike_time after multiplying by tau*(1-v))
        spikes = latency_encode(np.array([1.5, 0.0]), T=10, tau=5.0, strict=False)
        assert spikes.shape == (10, 2)
        # value=1.5 → spike_time = 5*(1-1.5) = -2.5 → clipped to 0
        assert bool(spikes[0, 0])
        # value=0 → spike_time = 5 → fires at index 5
        assert bool(spikes[5, 1])

    def test_strict_default_accepts_boundary_values(self):
        spikes = latency_encode(np.array([0.0, 1.0]), T=10, tau=5.0)
        assert spikes.shape == (10, 2)

    def test_strict_default_accepts_interior_values(self):
        spikes = latency_encode(np.array([0.25, 0.75]), T=20, tau=4.0)
        assert spikes.shape == (20, 2)
        # 0.75 should fire earlier than 0.25
        first_a = int(np.argmax(spikes[:, 0]))
        first_b = int(np.argmax(spikes[:, 1]))
        assert first_b < first_a

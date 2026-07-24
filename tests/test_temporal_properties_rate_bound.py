# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRateBound from former test_temporal_properties.py

"""Focused suite: TestRateBound from former test_temporal_properties.py."""

from __future__ import annotations

from tests.temporal_properties_support import *  # noqa: F403


class TestRateBound:
    """Sliding-window firing-rate safety bound checks."""

    def test_verified(self) -> None:
        """Sparse spikes remain below the configured rate bound."""
        s = _make_spikes()
        s[10, 0] = 1
        s[30, 0] = 1
        r = rate_bound(s, neuron_id=0, max_rate=0.5, window_size=10)
        assert r.result == PropertyResult.VERIFIED

    def test_violated(self) -> None:
        """A dense burst inside one window violates the rate bound."""
        s = _make_spikes()
        s[10:18, 0] = 1  # 8 spikes in 10-step window = rate 0.8
        r = rate_bound(s, neuron_id=0, max_rate=0.5, window_size=10)
        assert r.result == PropertyResult.VIOLATED

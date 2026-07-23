# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBoundedActivity from former test_temporal_properties.py

"""Focused suite: TestBoundedActivity from former test_temporal_properties.py."""

from __future__ import annotations

from tests.temporal_properties_support import *  # noqa: F403

class TestBoundedActivity:
    """Bounded total activity checks over neuron subsets."""

    def test_verified(self) -> None:
        """Activity within the total-spike bound verifies the property."""
        s = _make_spikes()
        s[10, 0] = 1
        s[20, 1] = 1
        r = bounded_activity(s, neuron_set=[0, 1, 2], window_size=10, max_total_spikes=3)
        assert r.result == PropertyResult.VERIFIED

    def test_violated(self) -> None:
        """A high-activity window violates the total-spike bound."""
        s = _make_spikes()
        s[10:15, 0] = 1
        s[10:15, 1] = 1
        r = bounded_activity(s, neuron_set=[0, 1], window_size=10, max_total_spikes=5)
        assert r.result == PropertyResult.VIOLATED

    def test_summary_pass(self) -> None:
        """Verified bounded-activity results include the pass status marker."""
        s = _make_spikes()
        r = bounded_activity(s, neuron_set=[0, 1], window_size=5, max_total_spikes=10)
        assert "PASS" in r.summary()

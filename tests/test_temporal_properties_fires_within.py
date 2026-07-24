# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFiresWithin from former test_temporal_properties.py

"""Focused suite: TestFiresWithin from former test_temporal_properties.py."""

from __future__ import annotations

from tests.temporal_properties_support import *  # noqa: F403


class TestFiresWithin:
    """Response-latency checks after explicit stimulus times."""

    def test_verified(self) -> None:
        """A response inside the latency window verifies the property."""
        s = _make_spikes()
        s[12, 0] = 1  # responds 2 steps after stimulus at t=10
        r = fires_within(s, neuron_id=0, stimulus_times=[10], max_latency=5)
        assert r.result == PropertyResult.VERIFIED

    def test_violated(self) -> None:
        """A missing response returns a counterexample at the stimulus time."""
        s = _make_spikes()
        # No response
        r = fires_within(s, neuron_id=0, stimulus_times=[10], max_latency=5)
        assert r.result == PropertyResult.VIOLATED
        assert r.counterexample is not None
        assert r.counterexample.timestep == 10

    def test_multiple_stimuli(self) -> None:
        """Multiple stimuli all require responses inside their latency windows."""
        s = _make_spikes()
        s[12, 0] = 1
        s[22, 0] = 1
        r = fires_within(s, neuron_id=0, stimulus_times=[10, 20], max_latency=5)
        assert r.result == PropertyResult.VERIFIED

    def test_summary(self) -> None:
        """Violation summaries include the fail status marker."""
        s = _make_spikes()
        r = fires_within(s, neuron_id=0, stimulus_times=[10], max_latency=5)
        assert "FAIL" in r.summary()

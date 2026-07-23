# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRefractoryGuarantee from former test_temporal_properties.py

"""Focused suite: TestRefractoryGuarantee from former test_temporal_properties.py."""

from __future__ import annotations

from tests.temporal_properties_support import *  # noqa: F403

class TestRefractoryGuarantee:
    """Minimum inter-spike interval checks for one neuron."""

    def test_verified(self) -> None:
        """Spikes separated by at least ``min_gap`` verify the property."""
        s = _make_spikes()
        s[10, 0] = 1
        s[20, 0] = 1
        r = refractory_guarantee(s, neuron_id=0, min_gap=5)
        assert r.result == PropertyResult.VERIFIED

    def test_violated(self) -> None:
        """A too-short inter-spike interval returns the first spike time."""
        s = _make_spikes()
        s[10, 0] = 1
        s[12, 0] = 1  # gap = 2 < min_gap = 5
        r = refractory_guarantee(s, neuron_id=0, min_gap=5)
        assert r.result == PropertyResult.VIOLATED
        assert r.counterexample is not None
        assert r.counterexample.timestep == 10

    def test_no_spikes(self) -> None:
        """Silent neurons vacuously satisfy the refractory guarantee."""
        s = _make_spikes()
        r = refractory_guarantee(s, neuron_id=0, min_gap=5)
        assert r.result == PropertyResult.VERIFIED

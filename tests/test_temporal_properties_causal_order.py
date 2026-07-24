# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCausalOrder from former test_temporal_properties.py

"""Focused suite: TestCausalOrder from former test_temporal_properties.py."""

from __future__ import annotations

from tests.temporal_properties_support import *  # noqa: F403


class TestCausalOrder:
    """Causal-order checks between source and target neuron spikes."""

    def test_verified(self) -> None:
        """A source spike before each target spike verifies causal order."""
        s = _make_spikes()
        s[8, 0] = 1  # A fires at t=8
        s[10, 1] = 1  # B fires at t=10 (within 5 steps of A)
        r = causal_order(s, neuron_a=0, neuron_b=1, max_delay=5)
        assert r.result == PropertyResult.VERIFIED

    def test_violated(self) -> None:
        """A target spike without a recent source spike violates causal order."""
        s = _make_spikes()
        s[10, 1] = 1  # B fires, A never fires
        r = causal_order(s, neuron_a=0, neuron_b=1, max_delay=5)
        assert r.result == PropertyResult.VIOLATED

    def test_no_b_spikes(self) -> None:
        """No target spikes make the implication vacuously true."""
        s = _make_spikes()
        s[5, 0] = 1  # A fires but B never does → vacuously true
        r = causal_order(s, neuron_a=0, neuron_b=1, max_delay=5)
        assert r.result == PropertyResult.VERIFIED

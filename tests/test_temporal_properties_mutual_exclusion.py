# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMutualExclusion from former test_temporal_properties.py

"""Focused suite: TestMutualExclusion from former test_temporal_properties.py."""

from __future__ import annotations

from tests.temporal_properties_support import *  # noqa: F403

class TestMutualExclusion:
    """Mutual-exclusion checks over neuron subsets."""

    def test_verified(self) -> None:
        """Separated spikes in the checked set satisfy mutual exclusion."""
        s = _make_spikes()
        s[5, 0] = 1
        s[10, 1] = 1
        r = mutual_exclusion(s, neuron_set=[0, 1])
        assert r.result == PropertyResult.VERIFIED

    def test_violated(self) -> None:
        """Co-firing neurons produce a counterexample with both neuron IDs."""
        s = _make_spikes()
        s[5, 0] = 1
        s[5, 1] = 1
        r = mutual_exclusion(s, neuron_set=[0, 1])
        assert r.result == PropertyResult.VIOLATED
        assert r.counterexample is not None
        assert r.counterexample.timestep == 5
        assert set(r.counterexample.neuron_ids) == {0, 1}

    def test_three_neurons(self) -> None:
        """The checked subset may include more than two neurons."""
        s = _make_spikes()
        s[5, 0] = 1
        s[5, 2] = 1
        r = mutual_exclusion(s, neuron_set=[0, 1, 2])
        assert r.result == PropertyResult.VIOLATED

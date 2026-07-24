# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAkidaAnalytical from former test_model_akida_neuron.py

"""Focused suite: TestAkidaAnalytical from former test_model_akida_neuron.py."""

from __future__ import annotations

from tests.model_akida_neuron_support import *  # noqa: F403


class TestAkidaAnalytical:
    def test_rank_order_decay_formula(self):
        """V += int(weight · modulation^rank). Rank increments per event."""
        n = AkidaNeuron()
        w = 50
        # Rank 0: scaled = int(50 * 0.75^0) = 50
        n.step(w)
        assert n.v == 50 and n._rank == 1
        # Rank 1: scaled = int(50 * 0.75^1) = int(37.5) = 37
        n.step(w)
        assert n.v == 50 + 37 and n._rank == 2
        # Rank 2: scaled = int(50 * 0.75^2) = int(28.125) = 28
        n.step(w)
        assert n.v == 50 + 37 + 28 and n._rank == 3

    def test_modulation_decay_sequence(self):
        """modulation^rank: 1.0, 0.75, 0.5625, 0.4219, ..."""
        m = 0.75
        expected = [m**k for k in range(5)]
        assert abs(expected[0] - 1.0) < 1e-12
        assert abs(expected[1] - 0.75) < 1e-12
        assert abs(expected[2] - 0.5625) < 1e-12

    def test_zero_weight_no_integration(self):
        """weight=0 → no integration, rank does not increment."""
        n = AkidaNeuron()
        n.step(0)
        assert n.v == 0 and n._rank == 0

    def test_rank_only_increments_on_nonzero(self):
        """Rank increments only when weight != 0."""
        n = AkidaNeuron()
        n.step(50)  # rank → 1
        n.step(0)  # rank stays 1
        n.step(50)  # rank → 2
        assert n._rank == 2

    def test_single_spike_only(self):
        """Once _spiked=True, neuron never fires again."""
        n = AkidaNeuron(threshold=50)
        # First spike
        n.step(60)
        assert n._spiked is True
        # All subsequent steps return 0
        for _ in range(100):
            assert n.step(60) == 0

    def test_spike_at_threshold(self):
        """Spike when V >= threshold and not already spiked."""
        n = AkidaNeuron(threshold=100)
        # Feed until threshold
        n.step(100)  # V = 100 → spike
        assert n._spiked is True

    def test_no_leak(self):
        """No leak between events — V persists."""
        n = AkidaNeuron()
        n.step(50)
        v_after = n.v
        n.step(0)  # zero input
        assert n.v == v_after  # unchanged

    def test_integer_truncation(self):
        """int() truncates toward zero."""
        n = AkidaNeuron()
        # weight=1, rank=1: int(1 * 0.75) = int(0.75) = 0
        n.step(10)  # rank=0: int(10*1.0) = 10
        v1 = n.v
        n.step(1)  # rank=1: int(1*0.75) = 0
        assert n.v == v1  # no change (scaled=0)

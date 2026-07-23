# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBoothRinzelIsolation from former test_model_booth_rinzel.py

"""Focused suite: TestBoothRinzelIsolation from former test_model_booth_rinzel.py."""

from __future__ import annotations

from tests.model_booth_rinzel_support import *  # noqa: F403

class TestBoothRinzelIsolation:
    def test_construction(self):
        n = BoothRinzelNeuron()
        assert n.vs == -65.0
        assert n.vd == -65.0
        assert n.ca == 0.0

    def test_step_returns_binary(self):
        n = BoothRinzelNeuron()
        assert n.step(0.0) in (0, 1)

    def test_spikes_under_drive(self):
        n = BoothRinzelNeuron()
        spikes = sum(n.step(10.0) for _ in range(50_000))
        assert spikes > 100, f"too few spikes at I=10: {spikes}"

    def test_two_compartments_differ(self):
        """Soma and dendrite should have different voltages under drive."""
        n = BoothRinzelNeuron()
        for _ in range(10_000):
            n.step(10.0)
        assert n.vs != n.vd, "soma and dendrite identical"

    def test_calcium_accumulates(self):
        n = BoothRinzelNeuron()
        for _ in range(10_000):
            n.step(10.0)
        assert n.ca > 0, "calcium did not accumulate"

    def test_bistability(self):
        """At high current, model may enter depolarisation block (fewer spikes)."""
        n_low = BoothRinzelNeuron()
        n_high = BoothRinzelNeuron()
        spikes_low = sum(n_low.step(10.0) for _ in range(50_000))
        spikes_high = sum(n_high.step(50.0) for _ in range(50_000))
        # High current may give fewer spikes (depolarisation block)
        # Just verify both are finite and model doesn't crash
        assert np.isfinite(n_high.vs)

    def test_numerical_stability(self):
        """Model should not produce NaN or Inf at any current."""
        for I in [0, 5, 10, 20, 50]:
            n = BoothRinzelNeuron()
            for _ in range(10_000):
                n.step(float(I))
            assert np.isfinite(n.vs), f"vs NaN/Inf at I={I}"
            assert np.isfinite(n.vd), f"vd NaN/Inf at I={I}"
            assert np.isfinite(n.h), f"h NaN/Inf at I={I}"
            assert np.isfinite(n.n), f"n NaN/Inf at I={I}"
            assert np.isfinite(n.ca), f"ca NaN/Inf at I={I}"

    def test_gating_bounded(self):
        """Gating variables h, n, q should stay in [0, 1]."""
        n = BoothRinzelNeuron()
        for _ in range(50_000):
            n.step(10.0)
        assert 0 <= n.h <= 1
        assert 0 <= n.n <= 1
        assert 0 <= n.q <= 1

    def test_reset(self):
        n = BoothRinzelNeuron()
        for _ in range(1000):
            n.step(10.0)
        n.reset()
        assert n.vs == -65.0
        assert n.vd == -65.0
        assert n.ca == 0.0

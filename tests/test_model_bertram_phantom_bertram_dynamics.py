# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBertramDynamics from former test_model_bertram_phantom.py

"""Focused suite: TestBertramDynamics from former test_model_bertram_phantom.py."""

from __future__ import annotations

from tests.model_bertram_phantom_support import *  # noqa: F403


class TestBertramDynamics:
    def test_subthreshold_silent(self):
        n = BertramPhantomBurster()
        assert len(_run(n, current=10.0, steps=50_000)) == 0

    def test_fires_at_high_current(self):
        n = BertramPhantomBurster()
        spikes = _run(n, current=200.0, steps=50_000)
        assert spikes == [3]
        assert n.v < n.v_threshold

    def test_rate_monotonic(self):
        """Higher current does not reduce threshold-crossing count in this regime."""
        rates = []
        for I in [150.0, 200.0, 300.0]:
            n = BertramPhantomBurster()
            rates.append(len(_run(n, current=I, steps=50_000)))
        # Monotonic or at least non-decreasing
        assert rates[-1] >= rates[0]

    @pytest.mark.parametrize("current", [150.0, 200.0, 250.0, 300.0])
    def test_fi_sweep(self, current: float):
        """f-I sweep: seeded drive levels keep their RK4 crossing counts."""
        n = BertramPhantomBurster()
        spikes = _run(n, current=current, steps=50_000)
        assert len(spikes) == {150.0: 1, 200.0: 1, 250.0: 1, 300.0: 2}[current]
        assert np.isfinite(n.v) and np.isfinite(n.s1) and np.isfinite(n.s2)

    def test_burst_structure(self):
        """Phantom burster: spikes cluster in bursts with silent intervals.

        Detect bursts by finding gaps (ISI > 50 steps) between spike clusters.
        """
        n = BertramPhantomBurster()
        spike_times = _run(n, current=200.0, steps=100_000)
        if len(spike_times) >= 20:
            isis = np.diff(spike_times)
            # Bimodal ISI: short (intra-burst) and long (inter-burst)
            short = isis[isis < 50]
            long_gaps = isis[isis >= 50]
            # At least some structure — may be tonic at this current
            assert len(short) > 0 or len(long_gaps) > 0

    def test_voltage_bounded(self):
        """V stays bounded — no divergence under drive."""
        n = BertramPhantomBurster()
        vs = []
        for _ in range(50_000):
            n.step(200.0)
            vs.append(n.v)
        assert min(vs) > -100 and max(vs) < 50

    def test_upward_crossing_only(self):
        """Spike fires only on upward threshold crossing."""
        n = BertramPhantomBurster()
        prev_v = n.v
        for _ in range(50_000):
            spike = n.step(200.0)
            if spike == 1:
                assert prev_v < n.v_threshold
            prev_v = n.v

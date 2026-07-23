# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestActivityMonitor from former test_sc_runtime.py

"""Focused suite: TestActivityMonitor from former test_sc_runtime.py."""

from __future__ import annotations

from sc_runtime_support import *  # noqa: F403

class TestActivityMonitor:
    def test_observe_returns_metrics(self):
        mon = ActivityMonitor()
        bs = np.ones(100, dtype=np.uint8)
        m = mon.observe(bs)
        assert "density" in m
        assert "scc" in m
        assert "ema_scc" in m
        assert "activity_zone" in m

    def test_density_all_ones(self):
        mon = ActivityMonitor()
        m = mon.observe(np.ones(100, dtype=np.uint8))
        assert m["density"] == 1.0

    def test_density_all_zeros(self):
        mon = ActivityMonitor()
        m = mon.observe(np.zeros(100, dtype=np.uint8))
        assert m["density"] == 0.0

    def test_scc_with_reference(self):
        mon = ActivityMonitor()
        a = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
        b = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
        m = mon.observe(a, reference=b)
        assert m["scc"] == pytest.approx(1.0)

    def test_drift_detection(self):
        mon = ActivityMonitor(drift_threshold=0.2)
        a = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        b = a.copy()
        for _ in range(50):
            m = mon.observe(a, reference=b)
        assert m["drift_detected"] is True

    def test_mean_density_accumulates(self):
        mon = ActivityMonitor()
        for _ in range(10):
            mon.observe(np.ones(100, dtype=np.uint8))
        assert mon.mean_density == pytest.approx(1.0)

    def test_activity_zone_tracking(self):
        mon = ActivityMonitor()
        mon.observe(np.zeros(100, dtype=np.uint8))
        assert mon.current_zone == ActivityZone.IDLE

    def test_activity_zone_burst(self):
        mon = ActivityMonitor()
        mon.observe(np.ones(100, dtype=np.uint8))
        assert mon.current_zone == ActivityZone.BURST

    def test_scc_zero_streams_hit_numerator_floor(self):
        # All-zero stream and reference give pa=pb=p_and=0, so the numerator
        # collapses to the |num|<eps floor and the coefficient is 0.
        mon = ActivityMonitor()
        m = mon.observe(np.zeros(8, dtype=np.uint8), reference=np.zeros(8, dtype=np.uint8))
        assert m["scc"] == 0.0

    def test_compute_scc_degenerate_denominator_returns_zero(self):
        # A non-binary input breaks the bitstream invariant p_and<=min(pa,pb):
        # for [1.5,0.5] (pa=1.0) the denominator min(pa,pb)-pa*pb is exactly 0
        # while the numerator stays positive, exercising the |denom|<eps floor.
        mon = ActivityMonitor()
        degenerate = np.array([1.5, 0.5], dtype=np.float64)
        assert mon._compute_scc(degenerate, degenerate) == 0.0

    def test_mean_scc_within_bounds(self):
        mon = ActivityMonitor()
        a = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
        mon.observe(a, reference=a)
        assert -1.0 <= mon.mean_scc <= 1.0

    def test_drift_active_property(self):
        mon = ActivityMonitor(drift_threshold=0.2)
        a = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        for _ in range(50):
            mon.observe(a, reference=a)
        assert mon.drift_active is True

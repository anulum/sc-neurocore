# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestProfileReport from former test_spike_profiler.py

"""Focused suite: TestProfileReport from former test_spike_profiler.py."""

from __future__ import annotations

from tests.spike_profiler_support import *  # noqa: F403

class TestProfileReport:
    def test_summary_format(self):
        p = SpikeProfiler()
        rng = np.random.RandomState(0)
        for _ in range(10):
            p.record_step("h", _random_spikes(8, 0.2, rng))
        r = p.report()
        s = r.summary()
        assert "SpikeProfiler Report" in s
        assert "h:" in s

    def test_has_critical(self):
        r = ProfileReport(
            pathologies=[
                Pathology(Severity.CRITICAL, "test", "l", "msg", "fix"),
            ]
        )
        assert r.has_critical is True

    def test_no_critical(self):
        r = ProfileReport(
            pathologies=[
                Pathology(Severity.WARNING, "test", "l", "msg", "fix"),
            ]
        )
        assert r.has_critical is False

    def test_empty_has_critical(self):
        r = ProfileReport()
        assert r.has_critical is False

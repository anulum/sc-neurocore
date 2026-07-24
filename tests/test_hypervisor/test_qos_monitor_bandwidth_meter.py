# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBandwidthMeter from former test_qos_monitor.py

"""Focused suite: TestBandwidthMeter from former test_qos_monitor.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent))
from qos_monitor_support import *  # noqa: F403


class TestBandwidthMeter:
    def test_record_and_throughput(self) -> None:
        bm = BandwidthMeter()
        bm.record("t0", 100, 1000)
        bm.record("t0", 200, 2000)
        assert bm.throughput("t0") > 0

    def test_zero_throughput(self) -> None:
        bm = BandwidthMeter()
        assert bm.throughput("t_none") == 0.0

    def test_throughput_single_record_returns_raw_count(self) -> None:
        # With only one sample there is no time span to divide by, so the
        # throughput is reported as the raw spike count.
        bm = BandwidthMeter()
        bm.record("t0", 100, 1000)
        assert bm.throughput("t0") == 100.0

    def test_exceeds_quota(self) -> None:
        bm = BandwidthMeter()
        bm.record("t0", 1000, 100)
        bm.record("t0", 1000, 200)
        assert bm.exceeds_quota("t0", 0.001) is True

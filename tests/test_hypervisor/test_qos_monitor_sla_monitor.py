# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSLAMonitor from former test_qos_monitor.py

"""Focused suite: TestSLAMonitor from former test_qos_monitor.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent))
from qos_monitor_support import *  # noqa: F403

class TestSLAMonitor:
    def test_latency_ok(self) -> None:
        mon = SLAMonitor()
        t = _tenant("t0")
        t.qos.max_latency_us = 1000.0
        assert mon.check_latency(t, 500.0, 100) is None

    def test_latency_violation(self) -> None:
        mon = SLAMonitor()
        t = _tenant("t0")
        t.qos.max_latency_us = 1000.0
        v = mon.check_latency(t, 1500.0, 100)
        assert v is not None
        assert v.metric == "latency"
        assert mon.total_violations == 1

    def test_bandwidth_ok(self) -> None:
        mon = SLAMonitor()
        t = _tenant("t0")
        t.qos.max_bandwidth_mbps = 100.0
        assert mon.check_bandwidth(t, 50.0, 10) is None

    def test_bandwidth_violation(self) -> None:
        mon = SLAMonitor()
        t = _tenant("t0")
        t.qos.max_bandwidth_mbps = 100.0
        v = mon.check_bandwidth(t, 200.0, 50)
        assert v is not None

    def test_violations_for(self) -> None:
        mon = SLAMonitor()
        t0 = _tenant("t0")
        t0.qos.max_latency_us = 10.0
        t1 = _tenant("t1")
        t1.qos.max_latency_us = 10.0
        mon.check_latency(t0, 100.0, 1)
        mon.check_latency(t1, 100.0, 2)
        assert len(mon.violations_for("t0")) == 1

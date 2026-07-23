# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hypervisor runtime QoS-monitor contracts

"""Verify throughput metering, SLA detection, and definition ownership."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from sc_neurocore.hypervisor import hypervisor as compatibility_surface
from sc_neurocore.hypervisor import qos_monitor as qos_owner
from sc_neurocore.hypervisor.qos_monitor import BandwidthMeter, SLAMonitor
from sc_neurocore.hypervisor.tenant import Tenant, TenantPriority


def _tenant(
    tid: str = "t0", name: str = "test", prio: TenantPriority = TenantPriority.NORMAL
) -> Tenant:
    return Tenant(tenant_id=tid, name=name, priority=prio)


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


def test_defensive_empty_history_has_zero_throughput() -> None:
    meter = BandwidthMeter()
    meter._counters["tenant"] = []
    meter._timestamps["tenant"] = []

    assert meter.throughput("tenant") == 0.0


def test_zero_cycle_span_is_clamped() -> None:
    meter = BandwidthMeter()
    meter.record("tenant", 2, 100)
    meter.record("tenant", 3, 100)

    assert meter.throughput("tenant") == 5.0


def test_throughput_uses_last_one_hundred_samples() -> None:
    meter = BandwidthMeter()
    for cycle in range(101):
        meter.record("tenant", 1, cycle)

    assert meter.throughput("tenant") == pytest.approx(100 / 99)


def test_historical_surface_reexports_owner_objects_without_wrappers() -> None:
    assert compatibility_surface.BandwidthMeter is qos_owner.BandwidthMeter
    assert compatibility_surface.SLAViolation is qos_owner.SLAViolation
    assert compatibility_surface.SLAMonitor is qos_owner.SLAMonitor


def test_qos_monitor_definitions_have_one_owner() -> None:
    facade_tree = ast.parse(Path(compatibility_surface.__file__).read_text(encoding="utf-8"))
    owner_tree = ast.parse(Path(qos_owner.__file__).read_text(encoding="utf-8"))

    facade_classes = {node.name for node in facade_tree.body if isinstance(node, ast.ClassDef)}
    owner_classes = {node.name for node in owner_tree.body if isinstance(node, ast.ClassDef)}
    owned_names = {"BandwidthMeter", "SLAViolation", "SLAMonitor"}

    assert facade_classes.isdisjoint(owned_names)
    assert owner_classes == owned_names

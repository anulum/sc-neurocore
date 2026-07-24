# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Module-level tests from former test_qos_monitor.py

"""Module-level tests from former test_qos_monitor.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent))
from qos_monitor_support import *  # noqa: F403


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

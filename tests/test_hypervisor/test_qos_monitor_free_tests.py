# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Module-level tests from former test_qos_monitor.py

"""Module-level tests from former test_qos_monitor.py."""

from __future__ import annotations

import ast
import pytest
from pathlib import (
    Path,
)
from sc_neurocore.hypervisor import (
    hypervisor as compatibility_surface,
    qos_monitor as qos_owner,
)
from sc_neurocore.hypervisor.qos_monitor import (
    BandwidthMeter,
)


def test_defensive_empty_history_has_zero_throughput() -> None:
    """An empty history gives no throughput."""
    meter = BandwidthMeter()
    meter._counters["tenant"] = []
    meter._timestamps["tenant"] = []

    assert meter.throughput("tenant") == 0.0


def test_zero_cycle_span_gives_zero() -> None:
    """A window of no cycles makes the last sample its own base, so the rate is 0.0 rather than a clamped count."""
    meter = BandwidthMeter(window_cycles=0)
    meter.record("tenant", 2, 100)
    meter.record("tenant", 3, 200)

    assert meter.throughput("tenant") == 0.0


def test_throughput_counts_the_samples_after_the_base() -> None:
    """One spike per cycle over 100 cycles is a rate of 1.0, the first sample being the base."""
    meter = BandwidthMeter()
    for cycle in range(101):
        meter.record("tenant", 1, cycle)

    assert meter.throughput("tenant") == pytest.approx(1.0)


def test_historical_surface_reexports_owner_objects_without_wrappers() -> None:
    """The facade re-exports the owner's classes."""
    assert compatibility_surface.BandwidthMeter is qos_owner.BandwidthMeter
    assert compatibility_surface.SLAViolation is qos_owner.SLAViolation
    assert compatibility_surface.SLAMonitor is qos_owner.SLAMonitor


def test_qos_monitor_definitions_have_one_owner() -> None:
    """The QoS classes are defined in the owner module only."""
    facade_tree = ast.parse(Path(compatibility_surface.__file__).read_text(encoding="utf-8"))
    owner_tree = ast.parse(Path(qos_owner.__file__).read_text(encoding="utf-8"))

    facade_classes = {node.name for node in facade_tree.body if isinstance(node, ast.ClassDef)}
    owner_classes = {node.name for node in owner_tree.body if isinstance(node, ast.ClassDef)}
    owned_names = {"BandwidthMeter", "SLAViolation", "SLAMonitor"}

    assert facade_classes.isdisjoint(owned_names)
    assert owner_classes == owned_names

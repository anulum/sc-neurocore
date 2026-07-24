# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDeviceTelemetry from former test_tinysc_ports.py

"""Focused suite: TestDeviceTelemetry from former test_tinysc_ports.py."""

from __future__ import annotations

from tinysc_ports_support import *  # noqa: F403


class TestDeviceTelemetry:
    def test_record(self):
        dt = DeviceTelemetry()
        dt.record("L0", 5, 16)
        dt.record("L0", 3, 16)
        assert dt.total_ticks == 2
        assert dt.total_spikes == 8
        layer = dt.get_layer("L0")
        assert layer.tick_count == 2

    def test_summary(self):
        dt = DeviceTelemetry()
        dt.record("L0", 10, 32)
        s = dt.summary()
        assert s["total_spikes"] == 10
        assert "L0" in s["layers"]

    def test_layer_rate_and_zero_neuron_utilization_path(self):
        dt = DeviceTelemetry()
        dt.record("L0", 6, 0)  # should not push utilization sample
        dt.record("L0", 2, 10)  # should push one utilization sample (20%)
        layer = dt.get_layer("L0")
        assert layer.lifetime_spike_rate == pytest.approx(4.0)
        assert layer.mean_spike_rate == pytest.approx(4.0)
        assert layer.mean_utilization == pytest.approx(20.0)

    def test_get_layer_is_idempotent_and_initialises_zero_rates(self):
        dt = DeviceTelemetry()
        first = dt.get_layer("L-new")
        second = dt.get_layer("L-new")
        assert first is second
        assert first.lifetime_spike_rate == 0.0
        assert first.mean_spike_rate == 0.0
        assert first.mean_utilization == 0.0

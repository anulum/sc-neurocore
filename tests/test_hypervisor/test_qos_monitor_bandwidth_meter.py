# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBandwidthMeter from former test_qos_monitor.py

"""Focused suite: TestBandwidthMeter from former test_qos_monitor.py."""

from __future__ import annotations

from sc_neurocore.hypervisor.qos_monitor import (
    BandwidthMeter,
)


class TestBandwidthMeter:
    """Record samples and read throughput and quota from the meter."""

    def test_record_and_throughput(self) -> None:
        """Two samples give a positive rate."""
        bm = BandwidthMeter()
        bm.record("t0", 100, 1000)
        bm.record("t0", 200, 2000)
        assert bm.throughput("t0") > 0

    def test_zero_throughput(self) -> None:
        """An unknown tenant has no throughput."""
        bm = BandwidthMeter()
        assert bm.throughput("t_none") == 0.0

    def test_throughput_single_record_is_zero(self) -> None:
        """With one sample there is no interval to divide by, so the rate is 0.0, not the raw count."""
        bm = BandwidthMeter()
        bm.record("t0", 100, 1000)
        assert bm.throughput("t0") == 0.0

    def test_exceeds_quota(self) -> None:
        """Ten spikes per cycle of one bit at 1 MHz is 10 Mbps, above 0.001 Mbps."""
        bm = BandwidthMeter()
        bm.record("t0", 1000, 100)
        bm.record("t0", 1000, 200)
        assert bm.exceeds_quota("t0", 0.001, bits_per_spike=1, clock_hz=1e6) is True

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFPGAResourceReport from former test_sustainability_profiler.py

"""Focused suite: TestFPGAResourceReport from former test_sustainability_profiler.py."""

from __future__ import annotations

from sc_neurocore.energy_accounting.sustainability_profiler import (
    FPGAResourceReport,
)


class TestFPGAResourceReport:
    """Power estimates of FPGA resource reports."""

    def test_dynamic_power_positive(self) -> None:
        """A design draws positive dynamic power."""
        r = FPGAResourceReport(luts=10000, ffs=5000, clock_mhz=100)
        assert r.dynamic_power_mw > 0

    def test_dynamic_power_increases_with_luts(self) -> None:
        """More LUTs draw more dynamic power."""
        a = FPGAResourceReport(luts=1000)
        b = FPGAResourceReport(luts=100000)
        assert b.dynamic_power_mw > a.dynamic_power_mw

    def test_dynamic_power_increases_with_toggle(self) -> None:
        """A higher toggle rate draws more dynamic power."""
        a = FPGAResourceReport(luts=10000, toggle_rate=0.1)
        b = FPGAResourceReport(luts=10000, toggle_rate=0.5)
        assert b.dynamic_power_mw > a.dynamic_power_mw

    def test_total_includes_static(self) -> None:
        """The total includes the static power."""
        r = FPGAResourceReport(luts=10000, static_power_mw=100)
        assert r.total_power_mw >= 100

    def test_zero_resources_zero_dynamic(self) -> None:
        """No resources draw no dynamic power."""
        r = FPGAResourceReport(luts=0, ffs=0, bram_kb=0, dsp_slices=0)
        assert r.dynamic_power_mw == 0.0

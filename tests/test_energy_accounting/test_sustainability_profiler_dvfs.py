# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDVFS from former test_sustainability_profiler.py

"""Focused suite: TestDVFS from former test_sustainability_profiler.py."""

from __future__ import annotations

from sc_neurocore.energy_accounting.sustainability_profiler import (
    FPGAResourceReport,
)


class TestDVFS:
    """Voltage and frequency scaling of FPGA power."""

    def test_scale_reduces_power(self) -> None:
        """Scaling down lowers the total power."""
        r = FPGAResourceReport(luts=10000, clock_mhz=200, voltage_v=1.0)
        scaled = r.scale_dvfs(clock_mhz=100, voltage_v=0.7)
        assert scaled.total_power_mw < r.total_power_mw

    def test_scale_preserves_resources(self) -> None:
        """Scaling keeps the resource counts."""
        r = FPGAResourceReport(luts=10000, ffs=5000)
        scaled = r.scale_dvfs(50, 0.6)
        assert scaled.luts == 10000
        assert scaled.ffs == 5000

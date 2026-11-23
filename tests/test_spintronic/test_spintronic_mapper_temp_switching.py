# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTempSwitching from former test_spintronic_mapper.py

"""Focused suite: TestTempSwitching from former test_spintronic_mapper.py."""

from __future__ import annotations

from sc_neurocore.spintronic.spintronic_mapper import (
    switching_current_vs_temperature,
    switching_time_vs_temperature,
)


class TestTempSwitching:
    """Temperature dependence of switching."""

    def test_current_decreases_with_temp(self) -> None:
        """A hotter device needs less switching current."""
        ic_cold = switching_current_vs_temperature(50.0, 40.0, 200.0)
        ic_hot = switching_current_vs_temperature(50.0, 40.0, 400.0)
        assert ic_cold > ic_hot

    def test_time_increases_with_temp(self) -> None:
        """A hotter device takes longer to switch."""
        t_cold = switching_time_vs_temperature(1.0, 200.0)
        t_hot = switching_time_vs_temperature(1.0, 400.0)
        assert t_hot > t_cold

    def test_current_degenerate_parameters_return_baseline(self) -> None:
        """Degenerate parameters return the baseline current."""
        # A non-positive stability barrier leaves the model undefined, so the
        # baseline critical current is returned unchanged.
        assert switching_current_vs_temperature(50.0, 0.0, 300.0) == 50.0

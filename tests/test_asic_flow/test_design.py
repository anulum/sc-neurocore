# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ASIC design parameter tests

"""Exercise ASIC geometry, clocks, and stochastic synthesis settings."""

from __future__ import annotations

from sc_neurocore.asic_flow.asic_flow import DesignParams, SCASICOptimisationConfig


class TestDesignParams:
    def test_clock_period(self) -> None:
        dp = DesignParams(target_frequency_mhz=100.0)
        assert abs(dp.clock_period_ns - 10.0) < 0.01

    def test_die_dimensions(self) -> None:
        dp = DesignParams(die_area_um=(0, 0, 1000, 800))
        assert dp.die_width_um == 1000
        assert dp.die_height_um == 800

    def test_core_area(self) -> None:
        dp = DesignParams(core_area_um=(20, 20, 480, 480))
        assert abs(dp.core_area_mm2 - 0.2116) < 0.001


def test_sc_optimisation_can_disable_optional_yosys_passes() -> None:
    """Disabled width reduction and sharing leave only deterministic cleanup."""
    config = SCASICOptimisationConfig(
        reduce_constant_widths=False,
        share_stochastic_counters=False,
    )

    assert config.yosys_passes() == ["opt_clean -purge"]

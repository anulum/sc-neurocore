# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDriftCompensation from former test_intelligence_drift_compensation.py

"""Focused suite: TestDriftCompensation from former test_intelligence_drift_compensation.py."""

from __future__ import annotations

from tests.intelligence_drift_compensation_support import *  # noqa: F403

class TestDriftCompensation:
    """Analog drift compensation controller."""

    def test_basic_compensator(self) -> None:
        from sc_neurocore.compiler.intelligence import (
            generate_drift_compensator,
        )

        d = generate_drift_compensator("sc_analog")
        assert "module sc_analog_drift_ctrl" in d.verilog_controller
        assert "endmodule" in d.verilog_controller
        assert d.refresh_interval_ms > 0
        assert d.compensation_method == "periodic_refresh"

    def test_fast_drift(self) -> None:
        from sc_neurocore.compiler.intelligence import (
            generate_drift_compensator,
        )

        d = generate_drift_compensator(
            "sc_rram",
            drift_rate_per_day=0.1,
            max_drift_tolerance=0.01,
        )
        # Should refresh very frequently
        assert d.refresh_interval_ms < 10_000_000

    def test_verilog_contains_counter(self) -> None:
        from sc_neurocore.compiler.intelligence import (
            generate_drift_compensator,
        )

        d = generate_drift_compensator("sc_mem")
        assert "counter" in d.verilog_controller
        assert "refresh_trigger" in d.verilog_controller
        assert "REFRESH_CYCLES" in d.verilog_controller

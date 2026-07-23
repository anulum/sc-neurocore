# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPowerEstimatorVcdBranches from former test_power_estimator.py

"""Focused suite: TestPowerEstimatorVcdBranches from former test_power_estimator.py."""

from __future__ import annotations

from tests.power_estimator_support import *  # noqa: F403

class TestPowerEstimatorVcdBranches:
    """Branch coverage for the VCD loader/parser and the zero-spike-rate
    energy fallback in ``estimate_power``."""

    # A one-bit scalar trace: a single 0->1 transition on code ``!`` between
    # two timestamps. Exercises the scalar value-change branch (``line[0] in
    # "01"``) and gives the file-loading tests a non-zero toggle to assert on.
    _SCALAR_VCD = "$var wire 1 ! clk $end\n$enddefinitions $end\n#0\n0!\n#10\n1!\n"

    def test_vcd_from_path_object(self, tmp_path: Path) -> None:
        """A ``Path`` activity trace is read from disk (the ``isinstance(..., Path)``
        branch of the loader)."""
        vcd_file = tmp_path / "trace.vcd"
        vcd_file.write_text(self._SCALAR_VCD, encoding="utf-8")
        p = estimate_power("", activity_vcd=vcd_file, vcd_time_units_per_cycle=10)
        assert p.toggle_rate > 0

    def test_vcd_from_string_path_on_disk(self, tmp_path: Path) -> None:
        """A string path (no inline ``$var`` marker) that exists on disk is
        loaded from the file rather than parsed as literal VCD text."""
        vcd_file = tmp_path / "trace2.vcd"
        vcd_file.write_text(self._SCALAR_VCD, encoding="utf-8")
        p = estimate_power("", activity_vcd=str(vcd_file), vcd_time_units_per_cycle=10)
        assert p.toggle_rate > 0

    def test_vcd_skips_blank_malformed_and_unknown_entries(self) -> None:
        """Blank lines, vector lines without a code, non-binary values, and
        unknown codes are all skipped; only the well-formed change counts."""
        messy_vcd = (
            "$var wire 4 v bus [3:0] $end\n"
            "$enddefinitions $end\n"
            "\n"  # blank line -> skipped
            "#0\n"
            "b1010\n"  # vector line with no code (parts != 2) -> skipped
            "bxx10 v\n"  # non-binary value bits -> skipped
            "b0000 zz\n"  # unknown code -> skipped
            "b0101 v\n"  # first well-formed value for v
            "#10\n"
            "b1010 v\n"  # transition -> four bit toggles
        )
        p = estimate_power("", activity_vcd=messy_vcd, vcd_time_units_per_cycle=10)
        assert p.toggle_rate > 0

    def test_vcd_with_no_value_changes_returns_zero(self) -> None:
        """A trace with declarations and timestamps but no value changes has no
        observed codes, so the parser returns zero activity."""
        vcd = "$var wire 4 v bus [3:0] $end\n$enddefinitions $end\n#0\n#10\n"
        p = estimate_power("", activity_vcd=vcd, vcd_time_units_per_cycle=10)
        assert p.toggle_rate == 0.0
        assert p.dynamic_mw == 0.0

    def test_vcd_without_timestamps_uses_single_cycle(self) -> None:
        """Value changes with no ``#`` timestamp lines fall back to a single
        cycle denominator rather than dividing by a zero span."""
        vcd = "$var wire 4 v bus [3:0] $end\n$enddefinitions $end\nb0101 v\nb1010 v\n"
        p = estimate_power("", activity_vcd=vcd, vcd_time_units_per_cycle=10)
        assert p.toggle_rate > 0

    def test_zero_spike_rate_yields_zero_energy_per_spike(self) -> None:
        """A non-positive spike rate cannot be inverted, so energy-per-spike
        collapses to zero instead of dividing by zero."""
        v = "reg signed [15:0] v_reg; wire signed [31:0] _mul0 = a * b;"
        p = estimate_power(v, spike_rate_hz=0.0)
        assert p.energy_per_spike_nj == 0.0

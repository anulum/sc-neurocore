# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBuildEquivalenceMiter from former test_equivalence_miter.py

"""Focused suite: TestBuildEquivalenceMiter from former test_equivalence_miter.py."""

from __future__ import annotations

import pytest
from sc_neurocore.compiler.equivalence_miter import (
    build_equivalence_miter,
    MiterPort,
)
from tests.equivalence_miter_support import (
    _lif_ports,
)


class TestBuildEquivalenceMiter:
    """Rendering the miter Verilog."""

    def test_instantiates_both_modules_with_shared_inputs(self) -> None:
        """The miter instantiates both modules and drives them from shared inputs."""
        miter = build_equivalence_miter("sc_lif_neuron", "sc_lif_reference", _lif_ports())
        assert "sc_lif_neuron" in miter
        assert "sc_lif_reference" in miter
        assert "dut (" in miter
        assert "ref_model (" in miter
        # Free (non-reset) inputs are exposed as top-level miter inputs.
        assert "input wire signed [15:0] leak_k" in miter
        # The clock is a top-level input; reset is derived, not exposed.
        assert "input wire clk" in miter
        assert "input wire rst_n" not in miter

    def test_reset_counter_holds_reset_then_releases(self) -> None:
        """A reset counter holds reset for the given cycles, then releases it."""
        miter = build_equivalence_miter("dut_mod", "ref_mod", _lif_ports(), reset_cycles=3)
        assert "localparam integer RESET_CYCLES = 3;" in miter
        assert "reg [7:0] rst_cnt = 0;" in miter
        assert "wire rst_n = (rst_cnt >= RESET_CYCLES);" in miter
        # No simulation-only constructs that would over-constrain the checker.
        assert "initial" not in miter
        assert "#5" not in miter

    def test_asserts_every_output_pair_after_reset(self) -> None:
        """Every output pair is asserted equal once reset is released."""
        miter = build_equivalence_miter("dut_mod", "ref_mod", _lif_ports())
        assert "assert(spike_out_dut == spike_out_ref);" in miter
        assert "assert(v_out_dut == v_out_ref);" in miter
        assert "if (rst_n) begin" in miter

    def test_parameter_overrides_applied_per_instance(self) -> None:
        """Parameter overrides are applied to each instance separately."""
        miter = build_equivalence_miter(
            "dut_mod",
            "ref_mod",
            _lif_ports(),
            dut_params={"DATA_WIDTH": 16, "REFRACTORY_PERIOD": 0},
            ref_params={"DATA_WIDTH": 16},
        )
        assert ".REFRACTORY_PERIOD(0)" in miter
        assert ".DATA_WIDTH(16)" in miter

    def test_output_comparison_wires_declared(self) -> None:
        """Each output gets a typed wire for each side of the comparison."""
        miter = build_equivalence_miter("dut_mod", "ref_mod", _lif_ports())
        assert "wire signed [15:0] v_out_dut" in miter
        assert "wire signed [15:0] v_out_ref" in miter
        assert "wire spike_out_dut" in miter

    def test_same_top_names_rejected(self) -> None:
        """Two modules with the same top name are refused."""
        with pytest.raises(ValueError, match="must differ"):
            build_equivalence_miter("same", "same", _lif_ports())

    def test_missing_clock_rejected(self) -> None:
        """A port list without the clock is refused."""
        ports = [p for p in _lif_ports() if p.name != "clk"]
        with pytest.raises(ValueError, match="clock port"):
            build_equivalence_miter("dut_mod", "ref_mod", ports)

    def test_missing_reset_rejected(self) -> None:
        """A port list without the reset is refused."""
        ports = [p for p in _lif_ports() if p.name != "rst_n"]
        with pytest.raises(ValueError, match="reset port"):
            build_equivalence_miter("dut_mod", "ref_mod", ports)

    def test_no_outputs_rejected(self) -> None:
        """A port list without an output is refused."""
        ports = [p for p in _lif_ports() if p.direction != "output"]
        with pytest.raises(ValueError, match="at least one output"):
            build_equivalence_miter("dut_mod", "ref_mod", ports)

    def test_zero_reset_cycles_rejected(self) -> None:
        """Zero reset cycles are refused."""
        with pytest.raises(ValueError, match="at least 1"):
            build_equivalence_miter("dut_mod", "ref_mod", _lif_ports(), reset_cycles=0)

    def test_custom_clock_and_reset_names(self) -> None:
        """Custom clock and reset port names drive the miter's clock and reset."""
        ports = [
            MiterPort("clock_i", 1, False, "input"),
            MiterPort("resetn_i", 1, False, "input"),
            MiterPort("q", 8, False, "output"),
        ]
        miter = build_equivalence_miter("d", "r", ports, clock="clock_i", reset_n="resetn_i")
        assert "input wire clock_i" in miter
        assert "wire resetn_i = (rst_cnt >= RESET_CYCLES);" in miter
        assert "always @(posedge clock_i)" in miter

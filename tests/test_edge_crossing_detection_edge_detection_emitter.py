# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEdgeDetectionEmitter from former test_edge_crossing_detection.py

"""Focused suite: TestEdgeDetectionEmitter from former test_edge_crossing_detection.py."""

from __future__ import annotations

from tests.edge_crossing_detection_support import *  # noqa: F403

class TestEdgeDetectionEmitter:
    """The emitted RTL mirrors the runner's edge/level decision."""

    def test_crossing_no_reset_rtl_declares_edge_register(self) -> None:
        """A crossing, non-resetting model emits the 1-bit ``_thr_prev`` edge history."""
        rtl = UniversalNeuron.from_dict(_fhn_schema("crossing")).to_verilog(module_name="fhn_edge")
        assert "reg _thr_prev;" in rtl
        assert "!_thr_prev" in rtl
        assert "_thr_prev <=" in rtl

    def test_level_model_rtl_has_no_edge_register(self) -> None:
        """A level model emits the unchanged datapath with no edge history register."""
        rtl = UniversalNeuron.from_dict(_fhn_schema("level")).to_verilog(module_name="fhn_level")
        assert "_thr_prev" not in rtl

    def test_reset_crossing_model_rtl_has_no_edge_register(self) -> None:
        """A crossing model with a reset stays on the level datapath (no edge register)."""
        neuron = EquationNeuron(
            equations={"v": "-(v - E_L) / tau_m + I"},
            parameters={"E_L": -65.0, "tau_m": 10.0},
            state={"v": -65.0},
            threshold="v >= -50.0",
            reset={"v": "-65.0"},
            detection="crossing",
        )
        rtl = compile_to_verilog(neuron, module_name="lif_reset_crossing")
        assert "_thr_prev" not in rtl

    def test_explicit_previous_state_crossing_reads_register_and_candidate(self) -> None:
        """The compiler maps ``theta_prev`` to the register, not wrapped next state."""
        rtl = UniversalNeuron.from_dict(_wrapped_phase_schema()).to_verilog(
            module_name="wrapped_phase"
        )

        assert "(theta_reg < P_THETA_THRESHOLD)" in rtl
        assert "P_THETA_THRESHOLD <= (theta_reg +" in rtl
        assert "_thr_prev" not in rtl

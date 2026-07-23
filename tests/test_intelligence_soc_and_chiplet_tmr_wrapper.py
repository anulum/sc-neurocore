# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTMRWrapper from former test_intelligence_soc_and_chiplet.py

"""Focused suite: TestTMRWrapper from former test_intelligence_soc_and_chiplet.py."""

from __future__ import annotations

from tests.intelligence_soc_and_chiplet_support import *  # noqa: F403

class TestTMRWrapper:
    """Triple Modular Redundancy wrapper generation."""

    def test_majority_voter_structure(self):
        from sc_neurocore.compiler.intelligence import generate_tmr_wrapper

        v = generate_tmr_wrapper("sc_lif", data_width=16)
        assert "module sc_lif_tmr" in v
        assert "endmodule" in v
        assert "inst_a" in v
        assert "inst_b" in v
        assert "inst_c" in v
        assert "seu_detected" in v

    def test_median_voter(self):
        from sc_neurocore.compiler.intelligence import generate_tmr_wrapper

        v = generate_tmr_wrapper("sc_hh", data_width=32, voter="median")
        assert "Median" in v
        assert "sc_hh_tmr" in v

    def test_multi_state_var(self):
        from sc_neurocore.compiler.intelligence import generate_tmr_wrapper

        v = generate_tmr_wrapper("sc_izh", state_vars=["v", "u"])
        assert "v_voted" in v
        assert "u_voted" in v
        assert "v_a" in v
        assert "u_c" in v

    def test_seu_detection_wires(self):
        from sc_neurocore.compiler.intelligence import generate_tmr_wrapper

        v = generate_tmr_wrapper("sc_lif")
        assert "(v_a != v_b)" in v
        assert "(spike_a != spike_b)" in v

    def test_tmr_references_inner_module(self):
        from sc_neurocore.compiler.intelligence import generate_tmr_wrapper

        v = generate_tmr_wrapper("sc_custom_neuron")
        assert "sc_custom_neuron inst_a" in v
        assert "sc_custom_neuron inst_b" in v
        assert "sc_custom_neuron inst_c" in v

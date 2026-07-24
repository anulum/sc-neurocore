# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFormalToCertification from former test_e2e_pipeline.py

"""Focused suite: TestFormalToCertification from former test_e2e_pipeline.py."""

from __future__ import annotations

from tests.e2e.e2e_pipeline_support import *  # noqa: F403


@pytest.mark.e2e
class TestFormalToCertification:
    """Formal verification → safety certification evidence chain."""

    def test_sva_matches_verilog_ports(self):
        """SVA variable names match what a compiled Verilog module would have."""
        from sc_neurocore.compiler.static_analysis import generate_sva

        sva = generate_sva(
            STATE_VARS_LIF,
            data_width=16,
            fraction=8,
            module_name="sc_lif_formal",
        )
        assert "v_reg" in sva
        assert "spike_out" in sva
        assert "I_t" in sva
        assert "sc_lif_formal" in sva

    def test_sby_references_correct_module(self):
        """SymbiYosys script references the correct module name."""
        from sc_neurocore.compiler.deployment import generate_sby_script

        # Actual API: generate_sby_script(module_name, *, sva_file=...)
        sby = generate_sby_script(
            "sc_lif_formal",
            sva_file="sc_lif_sva.sv",
        )
        assert "sc_lif_formal.v" in sby
        assert "sc_lif_sva.sv" in sby
        assert "sc_lif_formal" in sby

    def test_certification_with_items(self):
        """Certification evidence XML includes all items."""
        from sc_neurocore.compiler.deployment import (
            generate_certification_evidence,
            CertificationItem,
        )

        items = [
            CertificationItem(
                req_id="REQ-001",
                description="No overflow",
                design_ref="sc_lif.v",
                verification_ref="sc_lif_sva.sv",
                status="PASS",
            ),
            CertificationItem(
                req_id="REQ-002",
                description="Spike reachable",
                design_ref="sc_lif.v",
                verification_ref="tb_lif.py",
                status="PASS",
            ),
        ]
        xml = generate_certification_evidence(
            "sc_lif_cert",
            items,
            standard="do254",
            dal_level="DAL-A",
        )
        assert "sc_lif_cert" in xml
        assert "DO-254" in xml
        assert "DAL-A" in xml
        assert "REQ-001" in xml
        assert "REQ-002" in xml

    def test_full_formal_chain(self):
        """SVA → .sby → certification: all module names consistent."""
        from sc_neurocore.compiler.static_analysis import generate_sva
        from sc_neurocore.compiler.deployment import (
            generate_sby_script,
            generate_certification_evidence,
            CertificationItem,
        )

        module = "sc_hh_formal"

        sva = generate_sva(["v", "n", "m", "h"], data_width=32, fraction=16, module_name=module)
        sby = generate_sby_script(module)
        items = [
            CertificationItem(
                req_id="REQ-100",
                description="Bounded membrane",
                design_ref=f"{module}.v",
                verification_ref=f"{module}_sva.sv",
                status="PASS",
            )
        ]
        xml = generate_certification_evidence(
            module,
            items,
            standard="iec61508",
            dal_level="SIL-3",
        )

        # All three reference the same module
        assert module in sva
        assert module in sby
        assert module in xml

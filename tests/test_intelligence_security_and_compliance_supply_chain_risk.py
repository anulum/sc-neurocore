# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSupplyChainRisk from former test_intelligence_security_and_compliance.py

"""Focused suite: TestSupplyChainRisk from former test_intelligence_security_and_compliance.py."""

from __future__ import annotations


class TestSupplyChainRisk:
    """Assessing supply-chain risk."""

    def test_low_risk(self) -> None:
        """A common target is low risk under EAR99."""
        from sc_neurocore.compiler.intelligence import (
            score_supply_chain_risk,
        )

        r = score_supply_chain_risk("artix7")
        assert r.risk_score < 50
        assert r.export_control == "EAR99"

    def test_high_risk_biological(self) -> None:
        """A biological target is high risk as emerging tech."""
        from sc_neurocore.compiler.intelligence import (
            score_supply_chain_risk,
        )

        r = score_supply_chain_risk("finalspark_neuroplatform")
        assert r.risk_score >= 50
        assert "Emerging tech" in " ".join(r.risk_factors)

    def test_alternatives_exist(self) -> None:
        """Alternatives are suggested."""
        from sc_neurocore.compiler.intelligence import (
            score_supply_chain_risk,
        )

        r = score_supply_chain_risk("artix7")
        assert len(r.alternatives) > 0

    def test_itar_for_radiation_hardened_fpga(self) -> None:
        """A radiation-hardened FPGA part is flagged ITAR-controlled."""
        from sc_neurocore.compiler.intelligence import score_supply_chain_risk

        r = score_supply_chain_risk("bae_rad750")
        assert r.export_control == "ITAR"
        assert "ITAR" in " ".join(r.risk_factors)

    def test_export_controlled_superconducting(self) -> None:
        """A superconducting platform is flagged export-controlled emerging tech."""
        from sc_neurocore.compiler.intelligence import score_supply_chain_risk

        r = score_supply_chain_risk("josephson_jj")
        assert r.export_control == "EAR-controlled"
        assert "superconducting" in " ".join(r.risk_factors).lower()

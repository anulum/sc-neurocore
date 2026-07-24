# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCostEstimation from former test_dna_mapper.py

"""Focused suite: TestCostEstimation from former test_dna_mapper.py."""

from __future__ import annotations

from dna_mapper_support import *  # noqa: F403


class TestCostEstimation:
    """Oligo synthesis cost estimation."""

    def test_cost_positive(self, simple_and_circuit: DNACircuitDesign) -> None:
        cost = estimate_cost(simple_and_circuit)
        assert cost["total_cost_usd"] > 0
        assert cost["n_unique_oligos"] > 0

    def test_hplc_more_expensive(self, simple_and_circuit: DNACircuitDesign) -> None:
        standard = estimate_cost(simple_and_circuit, purification="standard")
        hplc = estimate_cost(simple_and_circuit, purification="hplc")
        assert hplc["total_cost_usd"] > standard["total_cost_usd"]

    def test_cost_per_strand_present(self, simple_and_circuit: DNACircuitDesign) -> None:
        cost = estimate_cost(simple_and_circuit)
        assert "strand_costs" in cost
        for sc in cost["strand_costs"]:
            assert "name" in sc
            assert "length" in sc
            assert "cost_usd" in sc

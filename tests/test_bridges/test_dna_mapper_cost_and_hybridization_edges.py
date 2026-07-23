# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCostAndHybridizationEdges from former test_dna_mapper.py

"""Focused suite: TestCostAndHybridizationEdges from former test_dna_mapper.py."""

from __future__ import annotations

from dna_mapper_support import *  # noqa: F403

class TestCostAndHybridizationEdges:
    """Boundary contracts for synthesis cost and strand interaction helpers."""

    def test_cross_hybridization_empty_substring_is_zero(self) -> None:
        assert CrossHybridizationChecker._longest_common_substring("", "ACGT") == 0

    def test_estimate_cost_counts_duplicate_sequences_once(self) -> None:
        design = DNACircuitDesign(
            name="duplicate_cost",
            input_strands=[
                DNAStrand(name="a", sequence="ACGTACGT"),
                DNAStrand(name="b", sequence="ACGTACGT"),
            ],
        )

        cost = estimate_cost(design, price_per_base_usd=1.0, fixed_per_oligo_usd=0.0)

        assert cost["n_unique_oligos"] == 1
        assert cost["total_cost_usd"] == 8.0

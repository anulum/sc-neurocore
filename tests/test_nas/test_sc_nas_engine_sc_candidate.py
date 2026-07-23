# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSCCandidate from former test_sc_nas_engine.py

"""Focused suite: TestSCCandidate from former test_sc_nas_engine.py."""

from __future__ import annotations

from sc_nas_engine_support import *  # noqa: F403

class TestSCCandidate:
    def test_evaluate_resources(self) -> None:
        c = SCCandidate(
            layers=[
                LayerConfig(32, NeuronType.LIF, 256, DecorrelationStrategy.LFSR),
                LayerConfig(64, NeuronType.ADEX, 512, DecorrelationStrategy.SOBOL),
            ]
        )
        c.evaluate_resources()
        assert c.total_luts > 0
        assert c.total_ffs > 0
        assert c.total_dsp > 0
        assert c.total_bram_kb > 0
        assert c.total_power_mw > 0

    def test_meets_budget_within_limits(self) -> None:
        c = SCCandidate(
            layers=[
                LayerConfig(16, NeuronType.LIF, 64, DecorrelationStrategy.LFSR),
            ]
        )
        budget = FPGAResourceBudget(max_luts=1_000_000)
        assert c.meets_budget(budget)

    def test_exceeds_budget(self) -> None:
        c = SCCandidate(
            layers=[
                LayerConfig(256, NeuronType.HH, 4096, DecorrelationStrategy.HYBRID),
            ]
            * 10
        )
        budget = FPGAResourceBudget(max_luts=100)
        assert not c.meets_budget(budget)

    def test_fingerprint_deterministic(self) -> None:
        c = SCCandidate(
            layers=[
                LayerConfig(32, NeuronType.LIF, 256, DecorrelationStrategy.LFSR),
            ]
        )
        assert c.fingerprint == c.fingerprint

    def test_fingerprint_differs_for_different_configs(self) -> None:
        a = SCCandidate(
            layers=[
                LayerConfig(32, NeuronType.LIF, 256, DecorrelationStrategy.LFSR),
            ]
        )
        b = SCCandidate(
            layers=[
                LayerConfig(64, NeuronType.ADEX, 512, DecorrelationStrategy.SOBOL),
            ]
        )
        assert a.fingerprint != b.fingerprint

    def test_dsp_budget_check(self) -> None:
        c = SCCandidate(
            layers=[
                LayerConfig(256, NeuronType.HH, 256, DecorrelationStrategy.LFSR),
            ]
        )
        budget = FPGAResourceBudget(max_dsp=10)
        assert not c.meets_budget(budget)

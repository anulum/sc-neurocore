# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSCFitnessEvaluator from former test_sc_nas_engine.py

"""Focused suite: TestSCFitnessEvaluator from former test_sc_nas_engine.py."""

from __future__ import annotations

from sc_nas_engine_support import *  # noqa: F403


class TestSCFitnessEvaluator:
    def test_longer_bitstreams_higher_accuracy(self) -> None:
        ev = SCFitnessEvaluator(seed=42)
        short = SCCandidate(
            layers=[
                LayerConfig(32, NeuronType.LIF, 64, DecorrelationStrategy.LFSR),
            ]
        )
        long = SCCandidate(
            layers=[
                LayerConfig(32, NeuronType.LIF, 4096, DecorrelationStrategy.LFSR),
            ]
        )
        acc_short = ev.evaluate(short)
        acc_long = ev.evaluate(long)
        assert acc_long > acc_short

    def test_sobol_decorrelation_bonus(self) -> None:
        ev = SCFitnessEvaluator(seed=42)
        lfsr = SCCandidate(
            layers=[
                LayerConfig(32, NeuronType.LIF, 256, DecorrelationStrategy.LFSR),
            ]
        )
        sobol = SCCandidate(
            layers=[
                LayerConfig(32, NeuronType.LIF, 256, DecorrelationStrategy.SOBOL),
            ]
        )
        assert ev.evaluate(sobol) > ev.evaluate(lfsr)

    def test_accuracy_bounded_0_1(self) -> None:
        ev = SCFitnessEvaluator(seed=42)
        c = SCCandidate(
            layers=[
                LayerConfig(32, NeuronType.LIF, 128, DecorrelationStrategy.LFSR),
            ]
        )
        acc = ev.evaluate(c)
        assert 0.0 <= acc <= 1.0

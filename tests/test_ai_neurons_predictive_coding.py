# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — PredictiveCodingNeuron tests

"""Focused contracts for PredictiveCodingNeuron."""

from sc_neurocore.neurons.models.ai_optimized import PredictiveCodingNeuron


def test_predictive_coding_fires_on_change():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    n = PredictiveCodingNeuron()
    for _ in range(200):
        n.step(1.0)
    spikes = sum(n.step(10.0) for _ in range(50))
    assert spikes > 0


def test_predictive_coding_silent_on_constant():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    n = PredictiveCodingNeuron()
    for _ in range(500):
        n.step(0.5)
    late = sum(n.step(0.5) for _ in range(100))
    assert late == 0


def test_predictive_coding_reset():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    n = PredictiveCodingNeuron()
    for _ in range(50):
        n.step(5.0)
    n.reset()
    assert n.v == 0.0
    assert n.pred == 0.0

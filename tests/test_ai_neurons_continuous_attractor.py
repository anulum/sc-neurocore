# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ContinuousAttractorNeuron tests

"""Focused contracts for ContinuousAttractorNeuron."""

from sc_neurocore.neurons.models.ai_optimized import ContinuousAttractorNeuron


def test_continuous_attractor_fires():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    n = ContinuousAttractorNeuron()
    total = sum(n.step(2.0) for _ in range(200))
    assert total > 0


def test_continuous_attractor_bump_forms():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    n = ContinuousAttractorNeuron()
    for _ in range(200):
        n.step(2.0)
    assert max(n.u) > 0.0


def test_continuous_attractor_reset():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    n = ContinuousAttractorNeuron()
    for _ in range(100):
        n.step(2.0)
    n.reset()
    assert all(x == 0.0 for x in n.u)

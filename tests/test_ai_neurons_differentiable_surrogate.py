# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — DifferentiableSurrogateNeuron tests

"""Focused contracts for DifferentiableSurrogateNeuron."""

from sc_neurocore.neurons.models.ai_optimized import DifferentiableSurrogateNeuron


def test_differentiable_surrogate_fires():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    n = DifferentiableSurrogateNeuron()
    total = sum(n.step(1.5) for _ in range(20))
    assert total > 0


def test_differentiable_surrogate_grad_positive():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    n = DifferentiableSurrogateNeuron()
    assert n.surrogate_grad() > 0.0


def test_differentiable_surrogate_reset():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    n = DifferentiableSurrogateNeuron()
    for _ in range(10):
        n.step(1.5)
    n.reset()
    assert n.v == 0.0

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — AttentionGatedNeuron tests

"""Focused contracts for AttentionGatedNeuron."""

from sc_neurocore.neurons.models.ai_optimized import AttentionGatedNeuron


def test_attention_gated_fires():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    n = AttentionGatedNeuron()
    total = sum(n.step(2.0) for _ in range(200))
    assert total > 0


def test_attention_gated_suppresses_low_input():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    n = AttentionGatedNeuron(w_key=-2.0)
    total = sum(n.step(0.1) for _ in range(200))
    assert total == 0


def test_attention_gated_reset():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    n = AttentionGatedNeuron()
    for _ in range(50):
        n.step(2.0)
    n.reset()
    assert n.v == 0.0

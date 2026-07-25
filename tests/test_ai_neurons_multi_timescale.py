# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — MultiTimescaleNeuron tests

"""Focused contracts for MultiTimescaleNeuron."""

from sc_neurocore.neurons.models.ai_optimized import MultiTimescaleNeuron


def test_multi_timescale_fires():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    n = MultiTimescaleNeuron()
    total = sum(n.step(2.0) for _ in range(200))
    assert total > 0


def test_multi_timescale_slow_accumulates():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    n = MultiTimescaleNeuron()
    for _ in range(500):
        n.step(2.0)
    assert n.v_slow > 0.0


def test_multi_timescale_reset():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    n = MultiTimescaleNeuron()
    for _ in range(100):
        n.step(2.0)
    n.reset()
    assert n.v_fast == 0.0
    assert n.v_medium == 0.0
    assert n.v_slow == 0.0

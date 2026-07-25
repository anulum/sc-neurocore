# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — MetaPlasticNeuron tests

"""Focused contracts for MetaPlasticNeuron."""

from sc_neurocore.neurons.models.ai_optimized import MetaPlasticNeuron


def test_meta_plastic_fires():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    n = MetaPlasticNeuron()
    total = sum(n.step(2.0) for _ in range(200))
    assert total > 0


def test_meta_plastic_adapts_lr():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    n = MetaPlasticNeuron()
    lr_before = n.meta_lr
    for _ in range(100):
        n.step(2.0)
        n.update_meta(1.0)
    assert abs(n.meta_lr - lr_before) > 1e-6


def test_meta_plastic_reset():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    n = MetaPlasticNeuron()
    for _ in range(100):
        n.step(2.0)
        n.update_meta(1.0)
    n.reset()
    assert n.v == 0.0
    assert n.error_trace == 0.0
    assert n.expected_reward == 0.0

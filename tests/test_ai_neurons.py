# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for AI-optimized neuron models

from sc_neurocore.neurons.models.ai_optimized import (
    AttentionGatedNeuron,
    CompositionalBindingNeuron,
    ContinuousAttractorNeuron,
    DifferentiableSurrogateNeuron,
    MetaPlasticNeuron,
    MultiTimescaleNeuron,
    PredictiveCodingNeuron,
    SelfReferentialNeuron,
)


# ── MultiTimescaleNeuron ──────────────────────────────────────────

def test_multi_timescale_fires():
    n = MultiTimescaleNeuron()
    total = sum(n.step(2.0) for _ in range(200))
    assert total > 0


def test_multi_timescale_slow_accumulates():
    n = MultiTimescaleNeuron()
    for _ in range(500):
        n.step(2.0)
    assert n.v_slow > 0.0


def test_multi_timescale_reset():
    n = MultiTimescaleNeuron()
    for _ in range(100):
        n.step(2.0)
    n.reset()
    assert n.v_fast == 0.0
    assert n.v_medium == 0.0
    assert n.v_slow == 0.0


# ── AttentionGatedNeuron ──────────────────────────────────────────

def test_attention_gated_fires():
    n = AttentionGatedNeuron()
    total = sum(n.step(2.0) for _ in range(200))
    assert total > 0


def test_attention_gated_suppresses_low_input():
    n = AttentionGatedNeuron(w_key=-2.0)
    total = sum(n.step(0.1) for _ in range(200))
    assert total == 0


def test_attention_gated_reset():
    n = AttentionGatedNeuron()
    for _ in range(50):
        n.step(2.0)
    n.reset()
    assert n.v == 0.0


# ── PredictiveCodingNeuron ────────────────────────────────────────

def test_predictive_coding_fires_on_change():
    n = PredictiveCodingNeuron()
    for _ in range(200):
        n.step(1.0)
    spikes = sum(n.step(10.0) for _ in range(50))
    assert spikes > 0


def test_predictive_coding_silent_on_constant():
    n = PredictiveCodingNeuron()
    for _ in range(500):
        n.step(0.5)
    late = sum(n.step(0.5) for _ in range(100))
    assert late == 0


def test_predictive_coding_reset():
    n = PredictiveCodingNeuron()
    for _ in range(50):
        n.step(5.0)
    n.reset()
    assert n.v == 0.0
    assert n.pred == 0.0


# ── SelfReferentialNeuron ─────────────────────────────────────────

def test_self_referential_fires():
    n = SelfReferentialNeuron()
    total = sum(n.step(2.0) for _ in range(200))
    assert total > 0


def test_self_referential_adapts():
    n = SelfReferentialNeuron()
    for _ in range(200):
        n.step(2.0)
    assert sum(n._history) > 0


def test_self_referential_reset():
    n = SelfReferentialNeuron()
    for _ in range(100):
        n.step(2.0)
    n.reset()
    assert n.v == 0.0
    assert sum(n._history) == 0


# ── CompositionalBindingNeuron ────────────────────────────────────

def test_compositional_binding_fires():
    n = CompositionalBindingNeuron()
    total = sum(n.step(2.0) for _ in range(200))
    assert total > 0


def test_compositional_binding_phase_advances():
    n = CompositionalBindingNeuron()
    for _ in range(100):
        n.step(1.0)
    assert n.phi > 0.0


def test_compositional_binding_reset():
    n = CompositionalBindingNeuron()
    for _ in range(100):
        n.step(2.0)
    n.reset()
    assert n.phi == 0.0
    assert n.amplitude == 0.0


# ── DifferentiableSurrogateNeuron ─────────────────────────────────

def test_differentiable_surrogate_fires():
    n = DifferentiableSurrogateNeuron()
    total = sum(n.step(1.5) for _ in range(20))
    assert total > 0


def test_differentiable_surrogate_grad_positive():
    n = DifferentiableSurrogateNeuron()
    assert n.surrogate_grad() > 0.0


def test_differentiable_surrogate_reset():
    n = DifferentiableSurrogateNeuron()
    for _ in range(10):
        n.step(1.5)
    n.reset()
    assert n.v == 0.0


# ── ContinuousAttractorNeuron ────────────────────────────────────

def test_continuous_attractor_fires():
    n = ContinuousAttractorNeuron()
    total = sum(n.step(2.0) for _ in range(200))
    assert total > 0


def test_continuous_attractor_bump_forms():
    n = ContinuousAttractorNeuron()
    for _ in range(200):
        n.step(2.0)
    assert max(n.u) > 0.0


def test_continuous_attractor_reset():
    n = ContinuousAttractorNeuron()
    for _ in range(100):
        n.step(2.0)
    n.reset()
    assert all(x == 0.0 for x in n.u)


# ── MetaPlasticNeuron ────────────────────────────────────────────

def test_meta_plastic_fires():
    n = MetaPlasticNeuron()
    total = sum(n.step(2.0) for _ in range(200))
    assert total > 0


def test_meta_plastic_adapts_lr():
    n = MetaPlasticNeuron()
    lr_before = n.meta_lr
    for _ in range(100):
        n.step(2.0)
        n.update_meta(1.0)
    assert abs(n.meta_lr - lr_before) > 1e-6


def test_meta_plastic_reset():
    n = MetaPlasticNeuron()
    for _ in range(100):
        n.step(2.0)
        n.update_meta(1.0)
    n.reset()
    assert n.v == 0.0
    assert n.error_trace == 0.0
    assert n.expected_reward == 0.0

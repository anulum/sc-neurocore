# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEscapeRateStochasticMechanism from former test_model_escape_rate.py

"""Focused suite: TestEscapeRateStochasticMechanism from former test_model_escape_rate.py."""

from __future__ import annotations

from tests.model_escape_rate_support import *  # noqa: F403

class TestEscapeRateStochasticMechanism:
    """Core: P(spike) = ρ₀·exp((V-θ)/Δu)·dt. Bernoulli trial each step."""

    def test_stochastic_spiking(self):
        n = EscapeRateNeuron()
        spikes = sum(n.step(40.0) for _ in range(50000))
        assert spikes > 100

    def test_two_runs_differ(self):
        """Distinct explicit seeds produce distinct reproducible spike trains."""
        n1 = EscapeRateNeuron(seed=1)
        n2 = EscapeRateNeuron(seed=2)
        t1 = [n1.step(40.0) for _ in range(1000)]
        t2 = [n2.step(40.0) for _ in range(1000)]
        assert t1 != t2

    def test_explicit_seed_reset_replays_the_complete_stream(self):
        first = EscapeRateNeuron(seed=0x1234)
        expected = [first.step(40.0) for _ in range(4096)]
        expected_state = (first.v, first.rng_state)
        first.reset()
        actual = [first.step(40.0) for _ in range(4096)]
        assert actual == expected
        assert (first.v, first.rng_state) == expected_state

    def test_zero_seed_uses_the_documented_nonzero_hardware_fallback(self):
        neuron = EscapeRateNeuron(seed=0)
        assert neuron.initial_seed == neuron.rng_state == DEFAULT_LFSR16_SEED

    def test_default_seed_matches_schema_and_hardware_reproducibility(self):
        first = EscapeRateNeuron()
        second = EscapeRateNeuron()
        assert first.initial_seed == second.initial_seed == DEFAULT_LFSR16_SEED
        assert [first.step(40.0) for _ in range(1024)] == [second.step(40.0) for _ in range(1024)]

    @pytest.mark.parametrize("seed", [True, -1, 0x10000, 1.5])
    def test_invalid_explicit_seed_is_rejected(self, seed: object):
        with pytest.raises(ValueError, match="seed"):
            EscapeRateNeuron(seed=cast(int | None, seed))

    def test_trial_sample_is_exactly_eight_primitive_advances(self):
        expected = DEFAULT_LFSR16_SEED
        for _ in range(LFSR16_ADVANCES_PER_TRIAL):
            expected = lfsr16_advance(expected)
        assert lfsr16_trial_sample(DEFAULT_LFSR16_SEED) == expected

    def test_probability_threshold_endpoints_and_interior_quantisation(self):
        assert probability_to_lfsr16_threshold(0.0) == 0
        assert probability_to_lfsr16_threshold(1.0) == 0x10000
        assert probability_to_lfsr16_threshold(0.25) == 16_384
        sampler = Lfsr16Threshold(DEFAULT_LFSR16_SEED)
        before = sampler.state
        sampler.trial(0.0)
        assert sampler.state != before

    def test_rate_increases_with_input(self):
        n_low = EscapeRateNeuron()
        n_high = EscapeRateNeuron()
        s_low = sum(n_low.step(20.0) for _ in range(50000))
        s_high = sum(n_high.step(40.0) for _ in range(50000))
        assert s_high > s_low

    def test_zero_input_silent(self):
        """V_ss = -70, far below theta=-50 → P(spike) ≈ 0."""
        n = EscapeRateNeuron()
        spikes = sum(n.step(0.0) for _ in range(50000))
        assert spikes == 0

    def test_escape_probability_uses_bounded_hazard_transform(self):
        """Finite-step escape probability is 1 - exp(-rho(V) dt), not clipped rho dt."""
        n = EscapeRateNeuron(v=-50.0, v_threshold=-50.0, rho_0=0.2, dt=2.0)
        expected = 1.0 - math.exp(-0.4)
        assert n._spike_probability(n.v_threshold) == pytest.approx(expected)

    def test_high_escape_rate_saturates_without_invalid_probability(self):
        n = EscapeRateNeuron(v=1000.0)
        assert n.step(0.0) == 1
        assert n.v == n.v_reset

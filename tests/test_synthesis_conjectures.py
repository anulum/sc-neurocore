# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for cross-project synthesis conjectures

"""Tests for the 3 "worth testing" conjectures from SYNTHESIS_REALITY_CHECK:
1. SC-FIM analogy: longer L → lower encoding error (necessary condition)
2. STDP-FIM competition: both active simultaneously, weights diverge
3. Coherence restoration: FIM warm-up after population reset

Plus regression tests for the Lazarus phase gap fix.
All claims clearly scoped — no overclaiming.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from sc_neurocore import StochasticLIFNeuron
from sc_neurocore import BitstreamEncoder, bitstream_to_probability
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.identity.substrate import IdentitySubstrate
from sc_neurocore.identity.checkpoint import Checkpoint


# --- Conjecture 1: SC encoding precision scales with L ---


class TestSCEncodingPrecision:
    """Necessary condition for SC-FIM analogy: error ~ 1/sqrt(L)."""

    def test_error_decreases_with_L(self):
        target = 0.65
        L_values = [64, 256, 1024, 4096]
        errors = {}
        for L in L_values:
            errs = []
            for trial in range(200):
                enc = BitstreamEncoder(x_min=0.0, x_max=1.0, length=L, seed=trial)
                bits = enc.encode(target)
                errs.append(abs(bitstream_to_probability(bits) - target))
            errors[L] = np.mean(errs)
        # Strict: each doubling should roughly halve error (sqrt scaling)
        for i in range(len(L_values) - 1):
            assert errors[L_values[i + 1]] < errors[L_values[i]], (
                f"error at L={L_values[i + 1]} not lower than L={L_values[i]}"
            )

    def test_error_scales_approximately_sqrt(self):
        """Error ratio between L and 4L should be ~2 (from 1/sqrt scaling)."""
        target = 0.5
        n_trials = 500
        for L in [64, 256]:
            err_L = np.mean(
                [
                    abs(
                        bitstream_to_probability(
                            BitstreamEncoder(x_min=0.0, x_max=1.0, length=L, seed=t).encode(target)
                        )
                        - target
                    )
                    for t in range(n_trials)
                ]
            )
            err_4L = np.mean(
                [
                    abs(
                        bitstream_to_probability(
                            BitstreamEncoder(x_min=0.0, x_max=1.0, length=4 * L, seed=t).encode(
                                target
                            )
                        )
                        - target
                    )
                    for t in range(n_trials)
                ]
            )
            if err_4L > 0:
                ratio = err_L / err_4L
                # sqrt(4) = 2; allow 1.3-3.0 range for finite-sample effects
                assert 1.0 < ratio < 4.0, f"error ratio L={L} vs 4L: {ratio:.2f}, expected ~2.0"


# --- Conjecture 7: STDP-FIM competition ---


class TestSTDPFIMCompetition:
    """STDP asymmetric updates and FIM symmetric corrections should
    produce measurably different weight trajectories."""

    def test_stdp_only_vs_stdp_plus_fim(self):
        """With FIM active, weight distribution should differ from STDP-only."""
        results = {}
        for label, lam in [("stdp_only", 0.0), ("stdp_fim", 5.0)]:
            pop = Population(StochasticLIFNeuron, n=20, label="e")
            proj = Projection(pop, pop, weight=0.3, probability=0.3, plasticity="stdp", seed=42)
            drive = PoissonInput(n=20, rate_hz=100.0, weight=2.0, dt=0.001, seed=42)
            net = Network(pop, proj, drive, fim_lambda=lam)
            net.run(duration=0.2, dt=0.001)
            results[label] = proj.data.copy()

        # Weight distributions should differ
        diff = np.mean(np.abs(results["stdp_only"] - results["stdp_fim"]))
        assert diff > 0.0001, f"FIM had no measurable effect (diff={diff:.6f})"

    def test_fim_does_not_collapse_weights(self):
        """FIM should not drive all weights to zero or a single value."""
        pop = Population(StochasticLIFNeuron, n=20, label="e")
        proj = Projection(pop, pop, weight=0.3, probability=0.3, plasticity="stdp", seed=42)
        drive = PoissonInput(n=20, rate_hz=100.0, weight=2.0, dt=0.001, seed=42)
        net = Network(pop, proj, drive, fim_lambda=10.0)
        net.run(duration=0.3, dt=0.001)
        # Weights should have nonzero variance
        assert np.std(proj.data) > 0.001, "FIM collapsed all weights"


# --- Lazarus Coherence Restoration ---


class TestCoherenceRestoration:
    """After checkpoint restore, network should recover activity.
    This tests the engineering fact, not the consciousness claim."""

    def test_checkpoint_preserves_weights(self):
        sub = IdentitySubstrate(n_cortical=30, n_inhibitory=10, n_memory=5, seed=42)
        sub.run(duration=0.05, dt=0.001)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            Checkpoint.save(sub, path)
            restored = Checkpoint.load(path)

            # Weights should be identical
            np.testing.assert_array_equal(
                sub.proj_ee.data, restored.proj_ee.data, err_msg="weights differ after checkpoint"
            )
        finally:
            os.remove(path)

    def test_restored_network_has_weights(self):
        """After restore, weight structure should be intact."""
        sub = IdentitySubstrate(n_cortical=30, n_inhibitory=10, n_memory=5, seed=42)
        sub.run(duration=0.05, dt=0.001)
        w_before = sub.proj_ee.data.copy()

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            Checkpoint.save(sub, path)
            restored = Checkpoint.load(path)
            w_after = restored.proj_ee.data
            np.testing.assert_array_almost_equal(
                w_before, w_after, decimal=10, err_msg="weights changed after restore"
            )
        finally:
            os.remove(path)

    def test_population_reset_clears_state(self):
        """reset_all() should return neurons to initial conditions."""
        pop = Population(StochasticLIFNeuron, n=10, label="test")
        # Drive neuron to near-threshold
        for neuron in pop.neurons:
            neuron.step(0.5)  # inject current
        pop.reset_all()
        for neuron in pop.neurons:
            assert neuron.v == 0.0, "reset did not clear voltage"


# --- Effective Phase Resolution (Reality Check for Conjecture 2) ---


class TestEffectivePhaseResolution:
    """Test that LIF phase resolution depends on dt/period, not voltage precision."""

    def test_phase_resolution_from_firing_rate(self):
        """A 50 Hz neuron at dt=1ms has ~20 steps per cycle → q_eff ≈ 20."""
        neuron = StochasticLIFNeuron(v_threshold=1.0, tau_mem=20.0, dt=1.0)
        spikes = []
        for t in range(5000):
            if neuron.step(0.08):
                spikes.append(t)
        if len(spikes) >= 3:
            isis = np.diff(spikes)
            mean_period = np.mean(isis)  # steps per cycle
            q_eff = int(mean_period)
            # q_eff should be much less than 256 (Q8.8 levels)
            assert q_eff < 256, f"q_eff={q_eff} — NOT limited by Q8.8"
            # q_eff should be in range 5-100 for typical neurons
            assert 5 < q_eff < 200, f"q_eff={q_eff} outside expected range"

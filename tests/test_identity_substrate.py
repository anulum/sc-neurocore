# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for identity continuity substrate

"""Tests for the identity continuity substrate (Phase 1)."""

from __future__ import annotations

import numpy as np

from sc_neurocore.identity import (
    IdentitySubstrate,
    TraceEncoder,
    StateDecoder,
    Checkpoint,
    DirectorController,
)

# Small network sizes for fast tests
N_CORTICAL = 20
N_INHIBITORY = 8
N_MEMORY = 5


def _make_substrate(seed=42):
    return IdentitySubstrate(N_CORTICAL, N_INHIBITORY, N_MEMORY, seed=seed)


# --- Substrate creation and basic run ---


class TestSubstrateCreation:
    def test_populations_created(self):
        sub = _make_substrate()
        assert sub.cortical.n == N_CORTICAL
        assert sub.inhibitory.n == N_INHIBITORY
        assert sub.memory.n == N_MEMORY

    def test_projections_exist(self):
        sub = _make_substrate()
        assert sub.proj_ee.data.size > 0
        assert sub.proj_ei.data.size > 0
        assert sub.proj_ie.data.size > 0
        assert sub.proj_em.data.size > 0
        assert sub.proj_me.data.size > 0

    def test_single_step(self):
        sub = _make_substrate()
        spikes = sub.step()
        assert spikes.shape == (N_CORTICAL,)
        assert spikes.dtype == np.int8
        assert sub._total_steps == 1

    def test_step_zero_pads_short_stimuli(self):
        # Stimuli narrower than the cortical population are zero-padded to the
        # full width rather than truncating the injection.
        sub = _make_substrate()
        spikes = sub.step(stimuli=np.ones(N_CORTICAL // 4))
        assert spikes.shape == (N_CORTICAL,)

    def test_run_returns_correct_shape(self):
        sub = _make_substrate()
        result = sub.run(duration=0.01, dt=0.001)
        assert result.shape == (10, N_CORTICAL)

    def test_run_with_stimuli_sequence(self):
        sub = _make_substrate()
        n_steps = 10
        stim = np.random.default_rng(0).uniform(5, 15, (n_steps, N_CORTICAL))
        result = sub.run(duration=0.01, dt=0.001, stimuli_sequence=stim)
        assert result.shape == (n_steps, N_CORTICAL)


# --- Encoder ---


class TestTraceEncoder:
    def test_encode_shape(self):
        enc = TraceEncoder(n_neurons=N_CORTICAL, hash_dims=16, seed=42)
        pattern = enc.encode("The cat sat on the mat.", duration_ms=50, dt=0.001)
        assert pattern.shape == (N_CORTICAL, 50)

    def test_encode_produces_spikes(self):
        enc = TraceEncoder(n_neurons=N_CORTICAL, hash_dims=16, seed=42)
        pattern = enc.encode("Reasoning about identity and memory.", duration_ms=100, dt=0.001)
        assert pattern.sum() > 0

    def test_encode_empty_text(self):
        enc = TraceEncoder(n_neurons=N_CORTICAL, hash_dims=16, seed=42)
        pattern = enc.encode("", duration_ms=50, dt=0.001)
        assert pattern.shape == (N_CORTICAL, 50)

    def test_encode_key_value(self):
        enc = TraceEncoder(n_neurons=N_CORTICAL, hash_dims=16, seed=42)
        pattern = enc.encode_key_value("decision", "use PCA for dimensionality reduction")
        assert pattern.shape[0] == N_CORTICAL
        assert pattern.shape[1] > 0

    def test_different_texts_produce_different_patterns(self):
        enc = TraceEncoder(n_neurons=N_CORTICAL, hash_dims=16, seed=42)
        p1 = enc.encode("Alpha bravo charlie.", duration_ms=50, dt=0.001)
        p2 = enc.encode("Delta echo foxtrot.", duration_ms=50, dt=0.001)
        assert not np.array_equal(p1, p2)


# --- Inject experience and STDP ---


class TestExperienceInjection:
    def test_inject_changes_weights(self):
        sub = _make_substrate()
        weights_before = sub.ee_weights.copy()
        sub.inject_experience("The system decided to increase inhibition based on rate analysis.")
        weights_after = sub.ee_weights
        assert not np.array_equal(weights_before, weights_after)

    def test_inject_increases_step_count(self):
        sub = _make_substrate()
        steps_before = sub._total_steps
        sub.inject_experience("Short trace.")
        assert sub._total_steps > steps_before


# --- State extraction ---


class TestStateExtraction:
    def test_extract_state_empty(self):
        sub = _make_substrate()
        state = sub.extract_state()
        assert "firing_rates" in state
        assert "dominant_patterns" in state
        assert state["total_steps"] == 0

    def test_extract_state_after_run(self):
        sub = _make_substrate()
        stim = np.random.default_rng(0).uniform(5, 15, (100, N_CORTICAL))
        sub.run(duration=0.1, dt=0.001, stimuli_sequence=stim)
        state = sub.extract_state()
        assert state["firing_rates"].shape[0] > 0
        assert state["total_steps"] == 100


# --- Decoder ---


class TestDecoder:
    def test_priming_context_dormant(self):
        sub = _make_substrate()
        dec = StateDecoder(sub)
        ctx = dec.generate_priming_context()
        assert "dormant" in ctx.lower() or "0 steps" in ctx

    def test_priming_context_after_activity(self):
        sub = _make_substrate()
        stim = np.random.default_rng(0).uniform(5, 15, (200, N_CORTICAL))
        sub.run(duration=0.2, dt=0.001, stimuli_sequence=stim)
        dec = StateDecoder(sub)
        ctx = dec.generate_priming_context()
        assert "active" in ctx.lower() or "steps" in ctx.lower()
        assert len(ctx) > 20

    def test_connectivity_signature_shape(self):
        sub = _make_substrate()
        stim = np.random.default_rng(0).uniform(5, 15, (200, N_CORTICAL))
        sub.run(duration=0.2, dt=0.001, stimuli_sequence=stim)
        dec = StateDecoder(sub)
        fc = dec.extract_connectivity_signature()
        assert fc.ndim == 2
        assert fc.shape[0] == fc.shape[1]


# --- Checkpoint save/load ---


class TestCheckpoint:
    def test_round_trip(self, tmp_path):
        sub = _make_substrate()
        stim = np.random.default_rng(0).uniform(5, 15, (50, N_CORTICAL))
        sub.run(duration=0.05, dt=0.001, stimuli_sequence=stim)

        path = str(tmp_path / "test_checkpoint.npz")
        Checkpoint.save(sub, path)
        restored = Checkpoint.load(path)

        assert restored.n_cortical == sub.n_cortical
        assert restored.n_inhibitory == sub.n_inhibitory
        assert restored.n_memory == sub.n_memory
        assert restored._total_steps == sub._total_steps
        np.testing.assert_array_almost_equal(restored.ee_weights, sub.ee_weights)
        assert len(restored.spike_history) == len(sub.spike_history)

    def test_merge_two_checkpoints(self, tmp_path):
        sub1 = _make_substrate(seed=42)
        sub1.run(duration=0.02, dt=0.001)
        p1 = str(tmp_path / "ckpt1.npz")
        Checkpoint.save(sub1, p1)

        sub2 = _make_substrate(seed=42)
        stim = np.random.default_rng(7).uniform(5, 15, (20, N_CORTICAL))
        sub2.run(duration=0.02, dt=0.001, stimuli_sequence=stim)
        p2 = str(tmp_path / "ckpt2.npz")
        Checkpoint.save(sub2, p2)

        merged = Checkpoint.merge([p1, p2])
        assert merged.n_cortical == N_CORTICAL
        assert merged._total_steps == sub1._total_steps + sub2._total_steps


# --- Director controller ---


class TestDirector:
    def test_monitor_returns_metrics(self):
        sub = _make_substrate()
        stim = np.random.default_rng(0).uniform(5, 15, (100, N_CORTICAL))
        sub.run(duration=0.1, dt=0.001, stimuli_sequence=stim)
        director = DirectorController(sub)
        metrics = director.monitor()
        assert "mean_rate" in metrics
        assert "cv" in metrics
        assert "fano" in metrics

    def test_diagnose_returns_list(self):
        sub = _make_substrate()
        director = DirectorController(sub)
        problems = director.diagnose()
        assert isinstance(problems, list)

    def test_correct_does_not_crash(self):
        sub = _make_substrate()
        stim = np.random.default_rng(0).uniform(5, 15, (100, N_CORTICAL))
        sub.run(duration=0.1, dt=0.001, stimuli_sequence=stim)
        director = DirectorController(sub)
        director.correct()

    def test_report_is_readable(self):
        sub = _make_substrate()
        stim = np.random.default_rng(0).uniform(5, 15, (100, N_CORTICAL))
        sub.run(duration=0.1, dt=0.001, stimuli_sequence=stim)
        director = DirectorController(sub)
        report = director.report()
        assert "Rate:" in report
        assert "CV:" in report
        assert "Diagnosis:" in report


# --- Health check ---


class TestHealthCheck:
    def test_health_check_initial(self):
        sub = _make_substrate()
        hc = sub.health_check()
        assert hc["is_healthy"] is True
        assert hc["mean_rate"] == 0.0

    def test_health_check_reports_zero_spectral_entropy_for_silent_substrate(self):
        # Once enough silent history accumulates, the population train carries no
        # spectral power, so the spectral entropy collapses to zero.
        sub = _make_substrate()
        for _ in range(110):
            sub.step()
        hc = sub.health_check()
        assert hc["spectral_entropy"] == 0.0

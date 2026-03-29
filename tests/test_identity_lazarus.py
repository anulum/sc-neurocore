# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for identity substrate Lazarus protocol

"""Tests for IdentitySubstrate, TraceEncoder, StateDecoder,
Checkpoint save/load/merge, DirectorController."""

from __future__ import annotations

import os
import tempfile

import numpy as np

from sc_neurocore.identity.substrate import IdentitySubstrate
from sc_neurocore.identity.encoder import TraceEncoder
from sc_neurocore.identity.decoder import StateDecoder
from sc_neurocore.identity.checkpoint import Checkpoint
from sc_neurocore.identity.director import DirectorController


class TestIdentitySubstrate:
    def test_creation(self):
        sub = IdentitySubstrate(n_cortical=50, n_inhibitory=20, n_memory=10, seed=42)
        assert sub.n_cortical == 50
        assert sub.n_inhibitory == 20
        assert sub.n_memory == 10

    def test_step_returns_array(self):
        sub = IdentitySubstrate(n_cortical=20, n_inhibitory=10, n_memory=5, seed=42)
        spikes = sub.step(dt=0.001)
        assert isinstance(spikes, np.ndarray)
        assert len(spikes) == 20

    def test_run_shape(self):
        sub = IdentitySubstrate(n_cortical=20, n_inhibitory=10, n_memory=5, seed=42)
        result = sub.run(duration=0.05, dt=0.001)
        assert result.shape[0] == 50  # 50 ms / 1 ms
        assert result.shape[1] == 20

    def test_extract_state_keys(self):
        sub = IdentitySubstrate(n_cortical=30, n_inhibitory=10, n_memory=5, seed=42)
        sub.run(duration=0.02, dt=0.001)
        state = sub.extract_state()
        assert isinstance(state, dict)
        assert "firing_rates" in state or "total_steps" in state

    def test_health_check(self):
        sub = IdentitySubstrate(n_cortical=30, n_inhibitory=10, n_memory=5, seed=42)
        sub.run(duration=0.05, dt=0.001)
        health = sub.health_check()
        assert isinstance(health, dict)
        assert "is_healthy" in health or "mean_rate" in health


class TestTraceEncoder:
    def test_encode_shape(self):
        enc = TraceEncoder(n_neurons=50, hash_dims=32, seed=42)
        pattern = enc.encode("test input text", duration_ms=100, dt=0.001)
        assert pattern.shape == (50, 100)

    def test_encode_binary(self):
        enc = TraceEncoder(n_neurons=50, seed=42)
        pattern = enc.encode("hello world", duration_ms=50, dt=0.001)
        assert set(np.unique(pattern)).issubset({0, 1})

    def test_different_texts_different_patterns(self):
        enc = TraceEncoder(n_neurons=100, seed=42)
        p1 = enc.encode("alpha beta gamma", duration_ms=100, dt=0.001)
        p2 = enc.encode("completely different content", duration_ms=100, dt=0.001)
        assert not np.array_equal(p1, p2)

    def test_deterministic(self):
        enc1 = TraceEncoder(n_neurons=50, seed=42)
        enc2 = TraceEncoder(n_neurons=50, seed=42)
        p1 = enc1.encode("same text", duration_ms=50, dt=0.001)
        p2 = enc2.encode("same text", duration_ms=50, dt=0.001)
        np.testing.assert_array_equal(p1, p2)

    def test_empty_text(self):
        enc = TraceEncoder(n_neurons=50, seed=42)
        pattern = enc.encode("", duration_ms=50, dt=0.001)
        assert pattern.shape == (50, 50)

    def test_encode_key_value(self):
        enc = TraceEncoder(n_neurons=100, seed=42)
        pattern = enc.encode_key_value("project", "sc-neurocore")
        assert isinstance(pattern, np.ndarray)
        assert pattern.shape[0] == 100


class TestCheckpoint:
    def test_save_load_roundtrip(self):
        sub = IdentitySubstrate(n_cortical=30, n_inhibitory=10, n_memory=5, seed=42)
        sub.run(duration=0.02, dt=0.001)
        state_before = sub.extract_state()

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            Checkpoint.save(sub, path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

            restored = Checkpoint.load(path)
            state_after = restored.extract_state()

            # Compare total_steps if available
            if "total_steps" in state_before and "total_steps" in state_after:
                assert state_before["total_steps"] == state_after["total_steps"]
        finally:
            os.remove(path)

    def test_merge_same_architecture(self):
        """Merge two checkpoints from same architecture, same seed."""
        sub1 = IdentitySubstrate(n_cortical=20, n_inhibitory=8, n_memory=4, seed=42)
        sub1.run(duration=0.01, dt=0.001)

        sub2 = IdentitySubstrate(n_cortical=20, n_inhibitory=8, n_memory=4, seed=42)
        sub2.run(duration=0.01, dt=0.001)

        tmpdir = tempfile.mkdtemp()
        p1 = os.path.join(tmpdir, "s1.npz")
        p2 = os.path.join(tmpdir, "s2.npz")

        try:
            Checkpoint.save(sub1, p1)
            Checkpoint.save(sub2, p2)

            merged = Checkpoint.merge([p1, p2])
            assert merged is not None
            merged_state = merged.extract_state()
            assert isinstance(merged_state, dict)
        finally:
            os.remove(p1)
            os.remove(p2)
            os.rmdir(tmpdir)

    def test_load_file_not_found(self):
        try:
            Checkpoint.load("/nonexistent/path.npz")
            raise AssertionError("should raise FileNotFoundError")
        except (FileNotFoundError, OSError):
            pass


class TestStateDecoder:
    def test_dominant_patterns_shape(self):
        sub = IdentitySubstrate(n_cortical=50, n_inhibitory=20, n_memory=10, seed=42)
        sub.run(duration=0.05, dt=0.001)
        dec = StateDecoder(sub)
        patterns = dec.extract_dominant_patterns(n_components=3)
        assert isinstance(patterns, np.ndarray)

    def test_attractor_states(self):
        sub = IdentitySubstrate(n_cortical=50, n_inhibitory=20, n_memory=10, seed=42)
        sub.run(duration=0.1, dt=0.001)
        dec = StateDecoder(sub)
        attractors = dec.extract_attractor_states(threshold=0.3)
        assert isinstance(attractors, list)

    def test_connectivity_signature(self):
        sub = IdentitySubstrate(n_cortical=30, n_inhibitory=10, n_memory=5, seed=42)
        sub.run(duration=0.05, dt=0.001)
        dec = StateDecoder(sub)
        conn = dec.extract_connectivity_signature()
        assert isinstance(conn, np.ndarray)

    def test_priming_context_is_string(self):
        sub = IdentitySubstrate(n_cortical=30, n_inhibitory=10, n_memory=5, seed=42)
        sub.run(duration=0.05, dt=0.001)
        dec = StateDecoder(sub)
        ctx = dec.generate_priming_context()
        assert isinstance(ctx, str)


class TestDirectorController:
    def test_monitor_returns_dict(self):
        sub = IdentitySubstrate(n_cortical=30, n_inhibitory=10, n_memory=5, seed=42)
        sub.run(duration=0.05, dt=0.001)
        director = DirectorController(sub)
        metrics = director.monitor()
        assert isinstance(metrics, dict)

    def test_diagnose_returns_list(self):
        sub = IdentitySubstrate(n_cortical=30, n_inhibitory=10, n_memory=5, seed=42)
        sub.run(duration=0.05, dt=0.001)
        director = DirectorController(sub)
        problems = director.diagnose()
        assert isinstance(problems, list)

    def test_report_returns_string(self):
        sub = IdentitySubstrate(n_cortical=30, n_inhibitory=10, n_memory=5, seed=42)
        sub.run(duration=0.05, dt=0.001)
        director = DirectorController(sub)
        report = director.report()
        assert isinstance(report, str)
        assert len(report) > 0

    def test_correct_does_not_crash(self):
        sub = IdentitySubstrate(n_cortical=30, n_inhibitory=10, n_memory=5, seed=42)
        sub.run(duration=0.05, dt=0.001)
        director = DirectorController(sub)
        director.correct()  # should not raise

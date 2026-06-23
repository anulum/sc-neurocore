# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Predictive Coding Tests

import numpy as np
import pytest

from sc_neurocore.neuro_symbolic.predictive_coding import (
    HYPERVECTOR_DIM,
    Hypervector,
    PredictiveCodingLayer,
    ReasoningTrace,
    SymbolEncoder,
    VerifiableInference,
    _pack,
    _unpack,
)


# ── Hypervector Tests ────────────────────────────────────────────────


class TestHypervector:
    def test_zeros_popcount(self):
        hv = Hypervector.zeros()
        assert hv.popcount() == 0
        assert hv.length == HYPERVECTOR_DIM

    def test_random_near_half_density(self):
        hv = Hypervector.random(0xDEAD)
        assert abs(hv.density() - 0.5) < 0.05

    def test_random_deterministic(self):
        a = Hypervector.random(42)
        b = Hypervector.random(42)
        assert np.array_equal(a.data, b.data)

    def test_random_different_seeds_orthogonal(self):
        a = Hypervector.random(1)
        b = Hypervector.random(2)
        assert abs(a.similarity(b)) < 0.1

    def test_bind_self_inverse(self):
        a = Hypervector.random(100)
        b = Hypervector.random(200)
        recovered = a.bind(b).bind(b)
        assert np.array_equal(a.data, recovered.data)

    def test_bind_dissimilar_to_inputs(self):
        a = Hypervector.random(10)
        b = Hypervector.random(20)
        c = a.bind(b)
        assert abs(c.similarity(a)) < 0.1
        assert abs(c.similarity(b)) < 0.1

    def test_permute_preserves_popcount(self):
        hv = Hypervector.random(333)
        permuted = hv.permute(7)
        assert permuted.popcount() == hv.popcount()

    def test_permute_changes_vector(self):
        hv = Hypervector.random(555)
        permuted = hv.permute(1)
        assert not np.array_equal(hv.data, permuted.data)

    def test_threshold_bundle_majority(self):
        a = Hypervector.random(10)
        b = Hypervector.random(20)
        c = Hypervector.random(30)
        bundled = Hypervector.threshold_bundle([a, b, c])
        assert bundled.similarity(a) > 0.2
        assert bundled.similarity(b) > 0.2
        assert bundled.similarity(c) > 0.2

    def test_threshold_bundle_single(self):
        a = Hypervector.random(42)
        bundled = Hypervector.threshold_bundle([a])
        assert np.array_equal(bundled.data, a.data)

    def test_hamming_self_zero(self):
        a = Hypervector.random(77)
        assert a.hamming_distance(a) < 1e-10

    def test_similarity_self_one(self):
        a = Hypervector.random(88)
        assert abs(a.similarity(a) - 1.0) < 1e-10

    def test_pack_unpack_roundtrip(self):
        hv = Hypervector.random(999)
        bits = _unpack(hv)
        repacked = _pack(bits, hv.length)
        assert np.array_equal(hv.data, repacked.data)


# ── Symbol Encoder Tests ─────────────────────────────────────────────


class TestSymbolEncoder:
    def test_deterministic(self):
        enc1 = SymbolEncoder(42)
        enc2 = SymbolEncoder(42)
        a = enc1.encode("hello")
        b = enc2.encode("hello")
        assert np.array_equal(a.data, b.data)

    def test_different_symbols_orthogonal(self):
        enc = SymbolEncoder(42)
        a = enc.encode("cat")
        b = enc.encode("dog")
        assert abs(a.similarity(b)) < 0.15

    def test_vocabulary_size(self):
        enc = SymbolEncoder(42)
        enc.encode("a")
        enc.encode("b")
        enc.encode("a")
        assert enc.vocabulary_size == 2

    def test_sequence_order_matters(self):
        enc = SymbolEncoder(42)
        ab = enc.encode_sequence(["A", "B"])
        ba = enc.encode_sequence(["B", "A"])
        assert abs(ab.similarity(ba)) < 0.2

    def test_sequence_single_symbol(self):
        enc = SymbolEncoder(42)
        single = enc.encode("X")
        seq = enc.encode_sequence(["X"])
        assert np.array_equal(single.data, seq.data)


# ── Predictive Coding Layer Tests ────────────────────────────────────


class TestPredictiveCodingLayer:
    def test_predict_shape(self):
        layer = PredictiveCodingLayer(input_dim=16, hidden_dim=8)
        pred = layer.predict()
        assert pred.shape == (16,)

    def test_error_shape(self):
        layer = PredictiveCodingLayer(input_dim=16, hidden_dim=8)
        obs = np.random.default_rng(0).normal(0, 0.5, 16).astype(np.float32)
        error = layer.compute_error(obs)
        assert error.shape == (16,)

    def test_update_reduces_error(self):
        rng = np.random.default_rng(42)
        layer = PredictiveCodingLayer(input_dim=8, hidden_dim=4, lr=0.05, seed=42)
        target = rng.normal(0, 0.3, 8).astype(np.float32)
        errors = []
        for _ in range(200):
            mae = layer.update(target)
            errors.append(mae)
        assert errors[-1] < errors[0], "error should decrease over iterations"

    def test_convergence_flag(self):
        layer = PredictiveCodingLayer(input_dim=4, hidden_dim=2, lr=0.05, seed=0)
        target = np.array([0.1, -0.1, 0.2, -0.2], dtype=np.float32)
        for _ in range(500):
            layer.update(target)
        assert layer.converged or layer.mean_recent_error < 0.1

    def test_precision_scaling(self):
        obs = np.ones(4, dtype=np.float32) * 0.5
        low = PredictiveCodingLayer(4, 2, precision=0.1, seed=0)
        high = PredictiveCodingLayer(4, 2, precision=10.0, seed=0)
        err_low = np.abs(low.compute_error(obs)).mean()
        err_high = np.abs(high.compute_error(obs)).mean()
        assert err_high > err_low, "higher precision should amplify errors"


# ── Reasoning Trace Tests ────────────────────────────────────────────


class TestReasoningTrace:
    def test_empty_trace(self):
        trace = ReasoningTrace()
        assert trace.length == 0
        assert trace.mean_confidence == 0.0
        assert not trace.is_complete

    def test_add_steps(self):
        trace = ReasoningTrace()
        trace.add("cat", "match", 0.8, 0.9)
        trace.add("dog", "match", 0.3, 0.4)
        assert trace.length == 2
        assert abs(trace.mean_confidence - 0.65) < 0.01

    def test_finalize_marks_complete(self):
        trace = ReasoningTrace(start_ns=1)
        trace.add("x", "op", 0.5, 0.5)
        trace.finalize()
        assert trace.is_complete
        assert trace.end_ns > 0

    def test_to_dict_structure(self):
        trace = ReasoningTrace(start_ns=1)
        trace.add("sym", "op", 0.7, 0.8)
        trace.finalize()
        d = trace.to_dict()
        assert "steps" in d
        assert d["length"] == 1
        assert d["complete"] is True
        assert d["steps"][0]["symbol"] == "sym"


# ── Verifiable Inference Tests ───────────────────────────────────────


class TestVerifiableInference:
    def _make_engine(self, symbols=("cat", "dog", "bird")):
        enc = SymbolEncoder(42)
        layer = PredictiveCodingLayer(input_dim=8, hidden_dim=4, seed=0)
        vi = VerifiableInference(enc, layer)
        vi.register_symbols(symbols)
        return vi

    def test_register_symbols(self):
        vi = self._make_engine()
        assert vi.num_symbols == 3

    def test_infer_returns_results_and_trace(self):
        vi = self._make_engine()
        obs = np.random.default_rng(0).normal(0, 0.5, 8).astype(np.float32)
        results, trace = vi.infer(obs, top_k=2)
        assert len(results) == 2
        assert trace.is_complete
        assert trace.length >= 2

    def test_infer_result_has_name_and_score(self):
        vi = self._make_engine()
        obs = np.zeros(8, dtype=np.float32)
        results, _ = vi.infer(obs, top_k=1)
        assert len(results) == 1
        name, score = results[0]
        assert isinstance(name, str)
        assert -1.0 <= score <= 1.0

    def test_infer_empty_library(self):
        enc = SymbolEncoder(42)
        layer = PredictiveCodingLayer(input_dim=8, hidden_dim=4, seed=0)
        vi = VerifiableInference(enc, layer)
        obs = np.zeros(8, dtype=np.float32)
        results, trace = vi.infer(obs)
        assert results == []
        assert trace.is_complete

    def test_trace_records_prediction_error(self):
        vi = self._make_engine()
        obs = np.ones(8, dtype=np.float32) * 0.3
        _, trace = vi.infer(obs, top_k=1)
        step_ops = [s.operation for s in trace.steps]
        assert "compute_error" in step_ops

    def test_trace_records_hamming_match(self):
        vi = self._make_engine()
        obs = np.ones(8, dtype=np.float32) * 0.1
        _, trace = vi.infer(obs, top_k=2)
        match_steps = [s for s in trace.steps if s.operation == "hamming_match"]
        assert len(match_steps) == 2

    def test_top_k_respects_k(self):
        vi = self._make_engine(("a", "b", "c", "d", "e"))
        obs = np.random.default_rng(7).normal(0, 0.5, 8).astype(np.float32)
        results, _ = vi.infer(obs, top_k=3)
        assert len(results) == 3


# ── Edge-branch coverage ─────────────────────────────────────────────


class TestPredictiveCodingEdgeBranches:
    """Cover the no-op permutation, the empty-collection guards, and the
    error-history accessors before any prediction error is recorded."""

    def test_permute_by_multiple_of_length_returns_identity_copy(self) -> None:
        """A rotation that is a whole multiple of the dimension is the identity,
        so it returns an independent copy rather than rolling the bits."""
        hv = Hypervector.random(0x1234)
        rotated = hv.permute(0)
        assert np.array_equal(rotated.data, hv.data)
        # The result is a fresh array, not an alias of the source buffer.
        assert rotated.data is not hv.data

    def test_threshold_bundle_rejects_empty_input(self) -> None:
        """A majority vote over no vectors is undefined."""
        with pytest.raises(ValueError, match="cannot bundle zero vectors"):
            Hypervector.threshold_bundle([])

    def test_encode_sequence_rejects_empty_sequence(self) -> None:
        """An empty symbol sequence has no hypervector encoding."""
        encoder = SymbolEncoder(base_seed=7)
        with pytest.raises(ValueError, match="cannot encode empty sequence"):
            encoder.encode_sequence([])

    def test_error_accessors_on_fresh_layer(self) -> None:
        """Before any error is recorded, the recent-error mean is zero and the
        layer is not yet considered converged."""
        layer = PredictiveCodingLayer(input_dim=4, hidden_dim=3, seed=1)
        assert layer.mean_recent_error == 0.0
        assert layer.converged is False

    def test_mean_recent_error_after_a_few_updates(self) -> None:
        """With a populated but short history the mean is the recorded error and
        convergence still reports False (fewer than ten samples)."""
        layer = PredictiveCodingLayer(input_dim=4, hidden_dim=3, seed=1)
        observation = np.ones(4, dtype=np.float32)
        for _ in range(3):
            layer.compute_error(observation)
        # mu starts at zero so the prediction is zero and each error is unity.
        assert layer.mean_recent_error == pytest.approx(1.0)
        assert layer.converged is False

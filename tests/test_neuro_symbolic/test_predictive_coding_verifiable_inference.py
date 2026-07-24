# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestVerifiableInference from former test_predictive_coding.py

"""Focused suite: TestVerifiableInference from former test_predictive_coding.py."""

from __future__ import annotations

from predictive_coding_support import *  # noqa: F403


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

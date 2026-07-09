# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# Tests for sc_neurocore.explain (SNN explainability)

from __future__ import annotations

import numpy as np

from sc_neurocore.explain import (
    SpikeAttributor,
    TemporalSaliency,
    CausalImportance,
    ExplanationResult,
)


def _make_spikes(T=20, N=8, rate=0.2, seed=42):
    rng = np.random.RandomState(seed)
    return (rng.random((T, N)) < rate).astype(np.int8)


class TestExplanationResult:
    def test_top_k(self):
        imp = np.zeros((10, 5))
        imp[3, 2] = 1.0
        imp[7, 4] = 0.5
        r = ExplanationResult(method="test", importance_map=imp)
        top = r.top_k(2)
        assert len(top) == 2
        assert top[0] == (3, 2, 1.0)
        assert top[1] == (7, 4, 0.5)

    def test_summary(self):
        imp = np.random.rand(10, 5)
        r = ExplanationResult(method="test", importance_map=imp)
        s = r.summary()
        assert "test" in s
        assert "importance" in s


class TestSpikeAttributor:
    def test_basic(self):
        spikes = _make_spikes()
        weights = [np.random.randn(4, 8) * 0.3]
        attr = SpikeAttributor(decay=0.9)
        result = attr.attribute(spikes, weights, output_neuron=0)
        assert result.importance_map.shape == (20, 8)
        assert result.importance_map.max() <= 1.0

    def test_later_spikes_more_important(self):
        spikes = np.zeros((10, 4), dtype=np.int8)
        spikes[2, 0] = 1
        spikes[8, 0] = 1
        weights = [np.ones((2, 4))]
        attr = SpikeAttributor(decay=0.9)
        result = attr.attribute(spikes, weights, output_neuron=0)
        # Later spike should have higher importance (less decay)
        assert result.importance_map[8, 0] > result.importance_map[2, 0]

    def test_multi_layer_weights(self):
        spikes = _make_spikes(N=4)
        weights = [np.random.randn(8, 4), np.random.randn(2, 8)]
        attr = SpikeAttributor()
        result = attr.attribute(spikes, weights, output_neuron=0)
        assert result.importance_map.shape == (20, 4)

    def test_zero_spikes(self):
        spikes = np.zeros((10, 4), dtype=np.int8)
        weights = [np.random.randn(2, 4)]
        attr = SpikeAttributor()
        result = attr.attribute(spikes, weights, output_neuron=0)
        assert result.importance_map.max() == 0.0


class TestTemporalSaliency:
    def _run_fn(self, spikes):
        return spikes.sum(axis=0).astype(np.float64)

    def test_basic(self):
        spikes = _make_spikes()
        sal = TemporalSaliency(run_fn=self._run_fn)
        result = sal.explain(spikes, output_neuron=0)
        assert result.importance_map.shape == (20, 8)
        assert result.method == "temporal_saliency"

    def test_important_spike_detected(self):
        spikes = np.zeros((10, 4), dtype=np.int8)
        spikes[5, 0] = 1  # Only spike for neuron 0

        def run_fn(s):
            return s.sum(axis=0).astype(np.float64)

        sal = TemporalSaliency(run_fn=run_fn)
        result = sal.explain(spikes, output_neuron=0)
        # Removing the only spike for neuron 0 causes max change
        assert result.importance_map[5, 0] > 0

    def test_no_spikes(self):
        spikes = np.zeros((10, 4), dtype=np.int8)
        sal = TemporalSaliency(run_fn=lambda s: s.sum(axis=0).astype(np.float64))
        result = sal.explain(spikes, output_neuron=0)
        assert result.importance_map.max() == 0.0

    def test_scalar_output(self):
        def scalar_fn(s):
            return np.float64(s.sum())

        spikes = _make_spikes(N=4)
        sal = TemporalSaliency(run_fn=scalar_fn)
        result = sal.explain(spikes, output_neuron=0)
        assert result.importance_map.shape[0] == 20


class TestCausalImportance:
    def _run_fn(self, spikes):
        return spikes.sum(axis=0).astype(np.float64)

    def test_basic(self):
        spikes = _make_spikes()
        ci = CausalImportance(run_fn=self._run_fn)
        result = ci.explain(spikes, output_neuron=0)
        assert result.importance_map.shape == (1, 8)
        assert result.method == "causal_importance"

    def test_active_neuron_important(self):
        spikes = np.zeros((10, 4), dtype=np.int8)
        spikes[:, 0] = 1  # neuron 0 fires every step

        ci = CausalImportance(run_fn=self._run_fn)
        result = ci.explain(spikes, output_neuron=0)
        # Silencing neuron 0 causes biggest change
        assert result.importance_map[0, 0] >= result.importance_map[0, 1]

    def test_silent_neuron_unimportant(self):
        spikes = np.zeros((10, 4), dtype=np.int8)
        spikes[:, 0] = 1

        ci = CausalImportance(run_fn=self._run_fn)
        result = ci.explain(spikes, output_neuron=0)
        # Neuron 2 never fires, silencing it changes nothing
        assert result.importance_map[0, 2] == 0.0

    def test_scalar_output(self):
        def scalar_fn(s):
            return np.float64(s.sum())

        spikes = _make_spikes(N=4)
        ci = CausalImportance(run_fn=scalar_fn)
        result = ci.explain(spikes, output_neuron=0)
        assert result.importance_map.shape == (1, 4)

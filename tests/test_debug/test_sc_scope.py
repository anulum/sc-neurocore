# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC Bitstream Oscilloscope Tests

import numpy as np

from sc_neurocore.debug.sc_scope import (
    AnalysisWindow,
    BitstreamSample,
    LayerErrorBudget,
    LiveAnalyzer,
    ScopeRenderer,
    ScopeSession,
    TransportBackend,
    TransportConfig,
    TransportType,
    TriggerCondition,
    TriggerEngine,
    TriggerType,
    compute_scc,
)


# ── helpers ──────────────────────────────────────────────────────────


def _sample(layer: int = 0, density: float = 0.5, n_words: int = 8) -> BitstreamSample:
    rng = np.random.default_rng(42 + layer)
    threshold = int(density * 0xFFFF_FFFF)
    words = rng.integers(0, 0xFFFF_FFFF, size=n_words, dtype=np.uint32)
    packed = np.where(words < threshold, np.uint32(0xFFFF_FFFF), np.uint32(0))
    return BitstreamSample(
        timestamp_ns=layer * 1000,
        layer_id=layer,
        neuron_id=0,
        words=packed,
    )


# ── Transport Tests ──────────────────────────────────────────────────


class TestTransport:
    def test_simulated_connect(self):
        cfg = TransportConfig(TransportType.SIMULATED)
        tb = TransportBackend(cfg)
        assert tb.connect() is True
        assert tb.is_connected is True
        tb.disconnect()
        assert tb.is_connected is False

    def test_simulated_read(self):
        cfg = TransportConfig(TransportType.SIMULATED)
        tb = TransportBackend(cfg)
        tb.connect()
        words = tb.read_bitstream(8, layer_id=0)
        assert words is not None
        assert len(words) == 8
        assert words.dtype == np.uint32

    def test_read_without_connect(self):
        cfg = TransportConfig(TransportType.SIMULATED)
        tb = TransportBackend(cfg)
        assert tb.read_bitstream(8) is None

    def test_bytes_counted(self):
        cfg = TransportConfig(TransportType.SIMULATED)
        tb = TransportBackend(cfg)
        tb.connect()
        tb.read_bitstream(16)
        assert tb.bytes_received == 64

    def test_multiple_reads(self):
        cfg = TransportConfig(TransportType.SIMULATED)
        tb = TransportBackend(cfg)
        tb.connect()
        r1 = tb.read_bitstream(4)
        r2 = tb.read_bitstream(4)
        assert not np.array_equal(r1, r2)


# ── BitstreamSample Tests ────────────────────────────────────────────


class TestBitstreamSample:
    def test_bit_length(self):
        s = _sample(n_words=4)
        assert s.bit_length == 128

    def test_popcount_all_ones(self):
        words = np.array([0xFFFF_FFFF, 0xFFFF_FFFF], dtype=np.uint32)
        s = BitstreamSample(0, 0, 0, words)
        assert s.popcount == 64

    def test_popcount_all_zeros(self):
        words = np.array([0, 0], dtype=np.uint32)
        s = BitstreamSample(0, 0, 0, words)
        assert s.popcount == 0

    def test_density_range(self):
        s = _sample(density=0.5)
        assert 0.0 <= s.density <= 1.0

    def test_effective_bits_zero(self):
        words = np.array([0xFFFF_FFFF] * 4, dtype=np.uint32)
        s = BitstreamSample(0, 0, 0, words)
        assert s.effective_bits == 0.0  # No entropy at p=1

    def test_effective_bits_half(self):
        rng = np.random.default_rng(42)
        words = rng.integers(0, 0xFFFF_FFFF, size=16, dtype=np.uint32)
        s = BitstreamSample(0, 0, 0, words)
        assert s.effective_bits > 0


# ── AnalysisWindow Tests ─────────────────────────────────────────────


class TestAnalysisWindow:
    def test_push_and_count(self):
        w = AnalysisWindow(window_size=10)
        for i in range(5):
            w.push(_sample(density=0.5))
        assert w.count == 5

    def test_window_overflow(self):
        w = AnalysisWindow(window_size=4)
        for i in range(10):
            w.push(_sample(density=0.5))
        assert w.count == 4

    def test_mean_density(self):
        w = AnalysisWindow(window_size=100)
        for _ in range(20):
            w.push(_sample(density=1.0))
        assert abs(w.mean_density - 1.0) < 0.01

    def test_std_density(self):
        w = AnalysisWindow(window_size=100)
        for _ in range(20):
            w.push(_sample(density=0.5))
        # All same density → std ≈ 0
        assert w.std_density < 0.01


# ── SCC Tests ────────────────────────────────────────────────────────


class TestSCC:
    def test_identical_bitstreams(self):
        words = np.array([0xAAAA_AAAA] * 4, dtype=np.uint32)
        scc = compute_scc(words, words)
        assert abs(scc - 1.0) < 0.01

    def test_empty_bitstreams(self):
        scc = compute_scc(np.array([], dtype=np.uint32), np.array([], dtype=np.uint32))
        assert scc == 0.0

    def test_scc_range(self):
        rng = np.random.default_rng(42)
        a = rng.integers(0, 0xFFFF_FFFF, size=16, dtype=np.uint32)
        b = rng.integers(0, 0xFFFF_FFFF, size=16, dtype=np.uint32)
        scc = compute_scc(a, b)
        assert -1.0 <= scc <= 1.0


# ── LiveAnalyzer Tests ───────────────────────────────────────────────


class TestLiveAnalyzer:
    def test_ingest(self):
        la = LiveAnalyzer(num_layers=2)
        la.ingest(_sample(layer=0))
        la.ingest(_sample(layer=1))
        assert la.total_samples == 2

    def test_layer_stats(self):
        la = LiveAnalyzer(num_layers=1)
        for _ in range(10):
            la.ingest(_sample(layer=0, density=0.5))
        stats = la.layer_stats(0)
        assert "mean_density" in stats
        assert stats["sample_count"] == 10

    def test_all_stats(self):
        la = LiveAnalyzer(num_layers=3)
        for lid in range(3):
            la.ingest(_sample(layer=lid))
        all_s = la.all_stats()
        assert len(all_s) == 3

    def test_unknown_layer(self):
        la = LiveAnalyzer(num_layers=1)
        assert la.layer_stats(99) == {}


# ── LayerErrorBudget Tests ───────────────────────────────────────────


class TestLayerErrorBudget:
    def test_within_tolerance(self):
        eb = LayerErrorBudget(0, expected_density=0.5, tolerance=0.1)
        assert eb.check(0.52) is True

    def test_outside_tolerance(self):
        eb = LayerErrorBudget(0, expected_density=0.5, tolerance=0.01)
        assert eb.check(0.7) is False

    def test_violations(self):
        eb = LayerErrorBudget(0, expected_density=0.5, tolerance=0.05)
        eb.check(0.5)  # OK
        eb.check(0.8)  # violation
        eb.check(0.5)  # OK
        assert eb.violations == 1

    def test_pass_rate(self):
        eb = LayerErrorBudget(0, expected_density=0.5, tolerance=0.05)
        for _ in range(9):
            eb.check(0.5)
        eb.check(0.9)  # 1 violation
        assert abs(eb.pass_rate - 0.9) < 0.01

    def test_mean_error(self):
        eb = LayerErrorBudget(0, expected_density=0.5, tolerance=0.1)
        eb.check(0.6)  # err=0.1
        eb.check(0.4)  # err=0.1
        assert abs(eb.mean_error - 0.1) < 0.01

    def test_max_error(self):
        eb = LayerErrorBudget(0, expected_density=0.5, tolerance=0.1)
        eb.check(0.5)
        eb.check(0.9)
        assert abs(eb.max_error - 0.4) < 0.01


# ── Trigger Engine Tests ─────────────────────────────────────────────


class TestTriggerEngine:
    def test_density_above(self):
        te = TriggerEngine()
        te.add_trigger(TriggerCondition(TriggerType.DENSITY_ABOVE, threshold=0.9, layer_id=0))
        # High density sample
        words = np.array([0xFFFF_FFFF] * 8, dtype=np.uint32)
        s = BitstreamSample(0, 0, 0, words)
        events = te.evaluate(s)
        assert len(events) == 1
        assert events[0].trigger_type == TriggerType.DENSITY_ABOVE

    def test_density_below(self):
        te = TriggerEngine()
        te.add_trigger(TriggerCondition(TriggerType.DENSITY_BELOW, threshold=0.1, layer_id=0))
        words = np.array([0] * 8, dtype=np.uint32)
        s = BitstreamSample(0, 0, 0, words)
        events = te.evaluate(s)
        assert len(events) == 1

    def test_no_trigger_when_disabled(self):
        te = TriggerEngine()
        te.add_trigger(TriggerCondition(TriggerType.DENSITY_ABOVE, enabled=False))
        words = np.array([0xFFFF_FFFF] * 8, dtype=np.uint32)
        s = BitstreamSample(0, 0, 0, words)
        assert len(te.evaluate(s)) == 0

    def test_wrong_layer_skipped(self):
        te = TriggerEngine()
        te.add_trigger(TriggerCondition(TriggerType.DENSITY_ABOVE, threshold=0.5, layer_id=1))
        words = np.array([0xFFFF_FFFF] * 8, dtype=np.uint32)
        s = BitstreamSample(0, layer_id=0, neuron_id=0, words=words)
        assert len(te.evaluate(s)) == 0

    def test_event_count(self):
        te = TriggerEngine()
        te.add_trigger(TriggerCondition(TriggerType.SPIKE_DETECTED, layer_id=0))
        rng = np.random.default_rng(42)
        for i in range(5):
            words = rng.integers(1, 0xFFFF_FFFF, size=4, dtype=np.uint32)
            s = BitstreamSample(i * 100, 0, 0, words)
            te.evaluate(s)
        assert te.event_count > 0

    def test_clear(self):
        te = TriggerEngine()
        te.add_trigger(TriggerCondition(TriggerType.SPIKE_DETECTED, layer_id=0))
        words = np.array([0xFFFF_FFFF] * 4, dtype=np.uint32)
        te.evaluate(BitstreamSample(0, 0, 0, words))
        te.clear()
        assert te.event_count == 0


# ── ScopeSession Tests ───────────────────────────────────────────────


class TestScopeSession:
    def _make_session(self, num_layers=2):
        cfg = TransportConfig(TransportType.SIMULATED)
        tb = TransportBackend(cfg)
        la = LiveAnalyzer(num_layers=num_layers)
        return ScopeSession(transport=tb, analyzer=la)

    def test_start_stop(self):
        s = self._make_session()
        assert s.start() is True
        assert s.is_running is True
        s.stop()
        assert s.is_running is False

    def test_capture_one(self):
        s = self._make_session()
        s.start()
        sample = s.capture_one(layer_id=0, num_words=8)
        assert sample is not None
        assert sample.layer_id == 0
        assert s.sample_count == 1
        s.stop()

    def test_capture_sweep(self):
        s = self._make_session(num_layers=3)
        s.start()
        samples = s.capture_sweep(num_layers=3)
        assert len(samples) == 3
        assert s.sample_count == 3
        s.stop()

    def test_error_budget_integration(self):
        s = self._make_session()
        s.add_error_budget(0, expected_density=0.3, tol=0.5)
        s.start()
        for _ in range(10):
            s.capture_one(layer_id=0)
        assert 0 in s.error_budgets
        assert len(s.error_budgets[0].history) == 10
        s.stop()

    def test_status(self):
        s = self._make_session()
        s.start()
        s.capture_one()
        st = s.status()
        assert st["running"] is True
        assert st["samples"] == 1
        assert st["bytes_received"] > 0
        s.stop()

    def test_capture_without_start(self):
        s = self._make_session()
        assert s.capture_one() is None


# ── Scope Renderer Tests ─────────────────────────────────────────────


class TestScopeRenderer:
    def test_density_bar(self):
        bar = ScopeRenderer.render_density_bar(0.5)
        assert "█" in bar
        assert "░" in bar
        assert "0.500" in bar

    def test_density_bar_empty(self):
        bar = ScopeRenderer.render_density_bar(0.0)
        assert "░" in bar

    def test_density_bar_full(self):
        bar = ScopeRenderer.render_density_bar(1.0)
        assert "█" in bar

    def test_layer_summary(self):
        stats = {"mean_density": 0.5, "mean_effective_bits": 128.0, "sample_count": 10}
        line = ScopeRenderer.render_layer_summary(0, stats)
        assert "L0" in line
        assert "eff=" in line

    def test_render_session(self):
        cfg = TransportConfig(TransportType.SIMULATED)
        tb = TransportBackend(cfg)
        la = LiveAnalyzer(num_layers=2)
        session = ScopeSession(transport=tb, analyzer=la)
        session.start()
        for _ in range(5):
            session.capture_sweep(num_layers=2)
        text = ScopeRenderer.render_session(session)
        assert "SC Bitstream Scope" in text
        assert "LIVE" in text
        assert "L0" in text
        session.stop()

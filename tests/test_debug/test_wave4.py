# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for ScDoctor, HIL client, and AER router

"""Comprehensive tests mirroring Rust and Go test suites."""

import pytest
from sc_neurocore.debug.sc_doctor import ScDoctor
from sc_neurocore.debug.hil_client import (
    SpikeEvent,
    SpikeRingBuffer,
    LayerAggregator,
    ErrorBudget,
    CorrelationWindow,
    PrecisionTracker,
    EventFilter,
    filter_events,
    TriggerCondition,
    TriggerLog,
    RateLimiter,
    check_health,
    export_csv,
    export_json,
)
from sc_neurocore.bridges.aer_router import (
    SpikePacket,
    AERRouter,
    PACKET_SIZE,
)


# ===== ScDoctor Tests (mirror dynamic_adaptation Rust tests) =====


class TestScDoctor:
    def test_new_defaults(self):
        d = ScDoctor(512, 0.90)
        assert d.current_bitstream_length == 512
        assert d.target_precision == 0.90
        assert not d.error_correction_enabled

    def test_hamming74_roundtrip_all_patterns(self):
        d = ScDoctor()
        d.error_correction_enabled = True
        for data in range(16):
            encoded = d.encode_ecc(data)
            decoded = d.decode_ecc(encoded)
            assert data == decoded, f"Roundtrip failed for {data:#06b}"

    def test_hamming74_single_bit_correction(self):
        d = ScDoctor()
        d.error_correction_enabled = True
        data = 0b1011
        encoded = d.encode_ecc(data)
        for bit in range(7):
            corrupted = encoded ^ (1 << bit)
            recovered = d.decode_ecc(corrupted)
            assert data == recovered, f"Failed to correct bit {bit}"

    def test_ecc_bypass_when_disabled(self):
        d = ScDoctor()
        assert d.encode_ecc(0b1011) == 0b1011
        assert d.decode_ecc(0b1111011) == 0b1011

    def test_adapt_high_correlation_doubles(self):
        d = ScDoctor(256)
        d.adapt(0.20)
        assert d.current_bitstream_length == 512

    def test_adapt_low_correlation_halves(self):
        d = ScDoctor(512)
        d.adapt(0.03)
        assert d.current_bitstream_length == 256

    def test_adapt_floor_at_256(self):
        d = ScDoctor(256)
        d.adapt(0.03)
        assert d.current_bitstream_length == 256

    def test_adapt_enables_ecc_above_2048(self):
        d = ScDoctor(1024)
        d.adapt(0.30)
        assert not d.error_correction_enabled
        d.adapt(0.30)
        assert d.error_correction_enabled

    def test_adapt_disables_ecc_on_halve(self):
        d = ScDoctor(4096)
        d.error_correction_enabled = True
        d.adapt(0.03)
        assert not d.error_correction_enabled
        assert d.current_bitstream_length == 2048

    def test_adapt_mid_correlation_no_change(self):
        d = ScDoctor(512)
        d.adapt(0.10)
        assert d.current_bitstream_length == 512

    def test_encoded_fits_7_bits(self):
        d = ScDoctor()
        d.error_correction_enabled = True
        for data in range(16):
            assert d.encode_ecc(data) < 128

    def test_all_zero_pattern(self):
        d = ScDoctor()
        d.error_correction_enabled = True
        assert d.encode_ecc(0b0000) == 0b0000000

    def test_rust_dispatch_paths(self, monkeypatch: pytest.MonkeyPatch):
        import sc_neurocore.debug.sc_doctor as sc_doctor_mod

        class _FakeRustDoctor:
            @staticmethod
            def py_sc_doctor_adapt(length: int, ecc: bool, corr: float):
                return (length + 16, True)

            @staticmethod
            def py_hamming74_encode(data: int):
                return data ^ 0b1111111

            @staticmethod
            def py_hamming74_decode(encoded: int):
                return encoded ^ 0b1111111

        monkeypatch.setattr(sc_doctor_mod, "_HAS_RUST_DOCTOR", True)
        monkeypatch.setattr(sc_doctor_mod, "_sdc", _FakeRustDoctor())

        d = sc_doctor_mod.ScDoctor(256)
        d.adapt(0.1)
        assert d.current_bitstream_length == 272
        assert d.error_correction_enabled is True
        d.error_correction_enabled = True
        assert d.decode_ecc(d.encode_ecc(0b0110)) == 0b0110


# ===== HIL Client Tests (mirror Go hil_debugger tests) =====


class TestSpikeRingBuffer:
    def test_push_and_snapshot(self):
        rb = SpikeRingBuffer(4)
        for i in range(3):
            rb.push(SpikeEvent(sequence=i))
        snap = rb.snapshot()
        assert len(snap) == 3
        assert snap[0].sequence == 0

    def test_overwrite_on_full(self):
        rb = SpikeRingBuffer(2)
        for i in range(5):
            rb.push(SpikeEvent(sequence=i))
        snap = rb.snapshot()
        assert len(snap) == 2
        assert snap[-1].sequence == 4

    def test_snapshot_limit(self):
        rb = SpikeRingBuffer(100)
        for i in range(50):
            rb.push(SpikeEvent(sequence=i))
        snap = rb.snapshot(5)
        assert len(snap) == 5


class TestLayerAggregator:
    def test_record_and_get(self):
        la = LayerAggregator()
        la.record(SpikeEvent(layer_id="L0", correlation=0.1, precision=0.95))
        la.record(SpikeEvent(layer_id="L0", correlation=0.3, precision=0.85))
        ls = la.get("L0")
        assert ls is not None
        assert ls["event_count"] == 2
        assert la.mean_correlation(ls) == pytest.approx(0.2)

    def test_missing_layer(self):
        la = LayerAggregator()
        assert la.get("missing") is None


class TestErrorBudget:
    def test_no_violation(self):
        eb = ErrorBudget(min_precision=0.90, max_correlation=0.20)
        assert not eb.check(SpikeEvent(precision=0.95, correlation=0.10))

    def test_precision_violation(self):
        eb = ErrorBudget(min_precision=0.90)
        assert eb.check(SpikeEvent(precision=0.85))
        assert eb.violations == 1

    def test_correlation_violation(self):
        eb = ErrorBudget(max_correlation=0.10)
        assert eb.check(SpikeEvent(correlation=0.15))


class TestCorrelationWindow:
    def test_mean(self):
        cw = CorrelationWindow(4)
        for v in [0.1, 0.2, 0.3, 0.4]:
            cw.add(v)
        assert cw.mean() == pytest.approx(0.25)

    def test_max(self):
        cw = CorrelationWindow(4)
        for v in [0.1, 0.5, 0.2]:
            cw.add(v)
        assert cw.max() == pytest.approx(0.5)

    def test_count(self):
        cw = CorrelationWindow(10)
        for _ in range(5):
            cw.add(1.0)
        assert cw.count == 5


class TestPrecisionTracker:
    def test_ema(self):
        pt = PrecisionTracker(alpha=0.5)
        pt.update(1.0)
        assert pt.ema == 1.0
        pt.update(0.0)
        assert pt.ema == pytest.approx(0.5)


class TestEventFilter:
    def test_layer_filter(self):
        f = EventFilter(layer_id="L1")
        assert f.match(SpikeEvent(layer_id="L1"))
        assert not f.match(SpikeEvent(layer_id="L2"))

    def test_neuron_range(self):
        f = EventFilter(has_neuron=True, min_neuron=10, max_neuron=20)
        assert f.match(SpikeEvent(neuron_id=15))
        assert not f.match(SpikeEvent(neuron_id=25))

    def test_filter_events(self):
        events = [SpikeEvent(layer_id="L0"), SpikeEvent(layer_id="L1")]
        result = filter_events(events, EventFilter(layer_id="L1"))
        assert len(result) == 1


class TestTrigger:
    def test_armed_trigger(self):
        tc = TriggerCondition(min_correlation=0.5, armed=True)
        assert tc.evaluate(SpikeEvent(correlation=0.6))
        assert not tc.evaluate(SpikeEvent(correlation=0.3))

    def test_disarmed(self):
        tc = TriggerCondition(min_correlation=0.5, armed=False)
        assert not tc.evaluate(SpikeEvent(correlation=0.9))

    def test_trigger_log(self):
        tl = TriggerLog()
        tl.fire(SpikeEvent(sequence=1))
        tl.fire(SpikeEvent(sequence=2))
        assert tl.count == 2


class TestRateLimiter:
    def test_allow(self):
        rl = RateLimiter(3)
        assert rl.allow()
        assert rl.allow()
        assert rl.allow()
        assert not rl.allow()

    def test_refill(self):
        rl = RateLimiter(2)
        rl.allow()
        rl.allow()
        rl.refill(1)
        assert rl.allow()
        assert not rl.allow()


class TestHealthCheck:
    def test_healthy(self):
        h = check_health(100, 10, 50, 1000)
        assert h.status == "healthy"
        assert h.events_per_sec == pytest.approx(10.0)

    def test_buffer_pressure(self):
        h = check_health(100, 10, 999, 1000)
        assert h.status == "buffer_pressure"


class TestExport:
    def test_csv(self):
        events = [SpikeEvent(timestamp=1, layer_id="L0", precision=0.95)]
        csv_str = export_csv(events)
        assert "timestamp" in csv_str
        assert "L0" in csv_str

    def test_json(self):
        events = [SpikeEvent(timestamp=1, layer_id="L0")]
        j = export_json(events)
        data = __import__("json").loads(j)
        assert len(data) == 1
        assert data[0]["layer_id"] == "L0"


# ===== AER Router Tests (mirror Go interconnect tests) =====


class TestSpikePacket:
    def test_encode_decode(self):
        p = SpikePacket(source_id=100, target_id=200, timestamp=12345, spike_len=64, sequence=42)
        data = p.encode()
        assert len(data) == PACKET_SIZE
        p2 = SpikePacket.decode(data)
        assert p2.source_id == 100
        assert p2.target_id == 200
        assert p2.sequence == 42


class TestAERRouter:
    def test_register(self):
        r = AERRouter()
        r.register_route(100, "127.0.0.1:9001")
        assert r.route_count == 1

    def test_unregister(self):
        r = AERRouter()
        r.register_route(100, "127.0.0.1:9001")
        r.unregister_route(100)
        assert r.route_count == 0

    def test_dispatch_unregistered(self):
        r = AERRouter()
        assert not r.dispatch_spike(SpikePacket(target_id=999))

    def test_dispatch_increments_stats(self):
        r = AERRouter()
        r.register_route(100, "127.0.0.1:9001")
        r.dispatch_spike(SpikePacket(target_id=100, sequence=1))
        assert r.total_sent == 1
        s = r.get_stats(100)
        assert s.dispatched == 1

    def test_ack_clears_pending(self):
        r = AERRouter()
        r.register_route(100, "127.0.0.1:9001")
        r.dispatch_spike(SpikePacket(target_id=100, sequence=42))
        assert r.pending_count == 1
        r.ack_received(42)
        assert r.pending_count == 0

    def test_multi_route(self):
        r = AERRouter()
        for nid, port in [(10, 9001), (20, 9002), (30, 9003)]:
            r.register_route(nid, f"127.0.0.1:{port}")
        assert r.route_count == 3
        for nid in [10, 20, 30]:
            assert r.dispatch_spike(SpikePacket(target_id=nid, sequence=nid))
        assert r.total_sent == 3


class TestHilClientBranchCoverage:
    """Edge and accessor branches across the HIL telemetry components."""

    def test_ring_buffer_head_and_capacity_accessors(self):
        rb = SpikeRingBuffer(capacity=4)
        assert rb.capacity == 4
        assert rb.head == 0
        rb.push(SpikeEvent(timestamp=1))
        rb.push(SpikeEvent(timestamp=2))
        assert rb.head == 2

    def test_layer_aggregator_all_returns_independent_copies(self):
        agg = LayerAggregator()
        agg.record(SpikeEvent(layer_id="L1", correlation=0.3, precision=0.9))
        snapshot = agg.all()
        assert set(snapshot) == {"L1"}
        # Mutating the snapshot must not bleed back into the aggregator state.
        snapshot["L1"]["event_count"] = 999
        assert agg.all()["L1"]["event_count"] == 1

    def test_layer_aggregator_means_handle_zero_event_count(self):
        empty = {"event_count": 0, "sum_correlation": 0.0, "sum_precision": 0.0}
        assert LayerAggregator.mean_correlation(empty) == 0.0
        assert LayerAggregator.mean_precision(empty) == 0.0

    def test_layer_aggregator_mean_precision_divides_when_populated(self):
        ls = {"event_count": 2, "sum_correlation": 1.0, "sum_precision": 1.4}
        assert LayerAggregator.mean_precision(ls) == pytest.approx(0.7)

    def test_correlation_window_mean_and_max_empty_return_zero(self):
        win = CorrelationWindow(size=8)
        assert win.mean() == 0.0
        assert win.max() == 0.0

    def test_trigger_condition_layer_mismatch_does_not_fire(self):
        trig = TriggerCondition(min_correlation=0.5, layer_id="L1")
        # Event belongs to a different layer → trigger must not fire.
        assert trig.evaluate(SpikeEvent(layer_id="L2", correlation=0.9)) is False

    def test_rate_limiter_available_reflects_remaining_tokens(self):
        rl = RateLimiter(capacity=2)
        assert rl.available == 2
        rl.allow()
        assert rl.available == 1

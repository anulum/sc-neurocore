# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC Runtime Adaptation Tests

import numpy as np
import pytest

from sc_neurocore.control.sc_runtime import (
    ActivityMonitor,
    ActivityZone,
    AdaptationPolicy,
    DecorrelatorType,
    ECCMode,
    HammingECC,
    RuntimeConfig,
    RuntimeReport,
    SCRuntimeEngine,
    SECDEC_ECC,
    classify_activity,
)


# ── classify_activity Tests ──────────────────────────────────────────


class TestClassifyActivity:
    def test_idle(self):
        assert classify_activity(0.005) == ActivityZone.IDLE

    def test_low(self):
        assert classify_activity(0.03) == ActivityZone.LOW

    def test_normal(self):
        assert classify_activity(0.3) == ActivityZone.NORMAL

    def test_high(self):
        assert classify_activity(0.8) == ActivityZone.HIGH

    def test_burst(self):
        assert classify_activity(0.99) == ActivityZone.BURST

    def test_boundary_idle_low(self):
        assert classify_activity(0.01) == ActivityZone.LOW

    def test_boundary_low_normal(self):
        assert classify_activity(0.05) == ActivityZone.NORMAL


# ── RuntimeConfig Tests ──────────────────────────────────────────────


class TestRuntimeConfig:
    def test_default_values(self):
        c = RuntimeConfig()
        assert c.bitstream_length == 256
        assert c.decorrelator == DecorrelatorType.LFSR
        assert not c.ecc_enabled

    def test_effective_length_no_ecc(self):
        c = RuntimeConfig(bitstream_length=256, ecc_enabled=False)
        assert c.effective_length == 256

    def test_effective_length_with_ecc(self):
        c = RuntimeConfig(bitstream_length=256, ecc_enabled=True, ecc_mode=ECCMode.HAMMING)
        assert c.effective_length == 256 + (256 // 4) * 3

    def test_effective_length_secded(self):
        c = RuntimeConfig(bitstream_length=256, ecc_enabled=True, ecc_mode=ECCMode.SECDED)
        assert c.effective_length == 256 + (256 // 4) * 4

    def test_effective_length_parity(self):
        c = RuntimeConfig(bitstream_length=256, ecc_enabled=True, ecc_mode=ECCMode.PARITY)
        assert c.effective_length == 256 + (256 // 8)

    def test_copy_independent(self):
        c = RuntimeConfig(bitstream_length=512)
        d = c.copy()
        d.bitstream_length = 1024
        assert c.bitstream_length == 512

    def test_copy_preserves_ecc_mode(self):
        c = RuntimeConfig(ecc_mode=ECCMode.SECDED)
        d = c.copy()
        assert d.ecc_mode == ECCMode.SECDED


# ── ActivityMonitor Tests ────────────────────────────────────────────


class TestActivityMonitor:
    def test_observe_returns_metrics(self):
        mon = ActivityMonitor()
        bs = np.ones(100, dtype=np.uint8)
        m = mon.observe(bs)
        assert "density" in m
        assert "scc" in m
        assert "ema_scc" in m
        assert "activity_zone" in m

    def test_density_all_ones(self):
        mon = ActivityMonitor()
        m = mon.observe(np.ones(100, dtype=np.uint8))
        assert m["density"] == 1.0

    def test_density_all_zeros(self):
        mon = ActivityMonitor()
        m = mon.observe(np.zeros(100, dtype=np.uint8))
        assert m["density"] == 0.0

    def test_scc_with_reference(self):
        mon = ActivityMonitor()
        a = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
        b = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
        m = mon.observe(a, reference=b)
        assert m["scc"] == pytest.approx(1.0)

    def test_drift_detection(self):
        mon = ActivityMonitor(drift_threshold=0.2)
        a = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        b = a.copy()
        for _ in range(50):
            m = mon.observe(a, reference=b)
        assert m["drift_detected"] is True

    def test_mean_density_accumulates(self):
        mon = ActivityMonitor()
        for _ in range(10):
            mon.observe(np.ones(100, dtype=np.uint8))
        assert mon.mean_density == pytest.approx(1.0)

    def test_activity_zone_tracking(self):
        mon = ActivityMonitor()
        mon.observe(np.zeros(100, dtype=np.uint8))
        assert mon.current_zone == ActivityZone.IDLE

    def test_activity_zone_burst(self):
        mon = ActivityMonitor()
        mon.observe(np.ones(100, dtype=np.uint8))
        assert mon.current_zone == ActivityZone.BURST

    def test_scc_zero_streams_hit_numerator_floor(self):
        # All-zero stream and reference give pa=pb=p_and=0, so the numerator
        # collapses to the |num|<eps floor and the coefficient is 0.
        mon = ActivityMonitor()
        m = mon.observe(np.zeros(8, dtype=np.uint8), reference=np.zeros(8, dtype=np.uint8))
        assert m["scc"] == 0.0

    def test_compute_scc_degenerate_denominator_returns_zero(self):
        # A non-binary input breaks the bitstream invariant p_and<=min(pa,pb):
        # for [1.5,0.5] (pa=1.0) the denominator min(pa,pb)-pa*pb is exactly 0
        # while the numerator stays positive, exercising the |denom|<eps floor.
        mon = ActivityMonitor()
        degenerate = np.array([1.5, 0.5], dtype=np.float64)
        assert mon._compute_scc(degenerate, degenerate) == 0.0

    def test_mean_scc_within_bounds(self):
        mon = ActivityMonitor()
        a = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
        mon.observe(a, reference=a)
        assert -1.0 <= mon.mean_scc <= 1.0

    def test_drift_active_property(self):
        mon = ActivityMonitor(drift_threshold=0.2)
        a = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        for _ in range(50):
            mon.observe(a, reference=a)
        assert mon.drift_active is True


# ── HammingECC Tests ────────────────────────────────────────────────


class TestHammingECC:
    def test_roundtrip_all_patterns(self):
        ecc = HammingECC()
        for data in range(16):
            encoded = ecc.encode(data)
            decoded = ecc.decode(encoded)
            assert decoded == data, f"Roundtrip failed for {data}"

    def test_single_bit_correction(self):
        ecc = HammingECC()
        data = 0b1011
        encoded = ecc.encode(data)
        for bit in range(7):
            corrupted = encoded ^ (1 << bit)
            recovered = ecc.decode(corrupted)
            assert recovered == data, f"Failed to correct bit {bit}"

    def test_encoded_fits_7_bits(self):
        ecc = HammingECC()
        for data in range(16):
            assert ecc.encode(data) < 128

    def test_bitstream_roundtrip(self):
        ecc = HammingECC()
        bs = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        encoded = ecc.encode_bitstream(bs)
        decoded = ecc.decode_bitstream(encoded)
        np.testing.assert_array_equal(decoded[: len(bs)], bs)

    def test_bitstream_ecc_detects_corruption(self):
        ecc = HammingECC()
        bs = np.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=np.uint8)
        encoded = ecc.encode_bitstream(bs)
        encoded[3] ^= 1
        decoded = ecc.decode_bitstream(encoded)
        np.testing.assert_array_equal(decoded[: len(bs)], bs)


# ── SECDED Tests ────────────────────────────────────────────────────


class TestSECDED:
    def test_roundtrip_all_patterns(self):
        ecc = SECDEC_ECC()
        for data in range(16):
            encoded = ecc.encode(data)
            decoded, uncorrectable = ecc.decode(encoded)
            assert decoded == data
            assert not uncorrectable

    def test_encoded_fits_8_bits(self):
        ecc = SECDEC_ECC()
        for data in range(16):
            assert ecc.encode(data) < 256

    def test_single_bit_correction(self):
        ecc = SECDEC_ECC()
        for data in range(16):
            encoded = ecc.encode(data)
            for bit in range(8):
                corrupted = encoded ^ (1 << bit)
                decoded, uncorrectable = ecc.decode(corrupted)
                assert decoded == data, f"Failed 1-bit correction for data={data}, bit={bit}"
                assert not uncorrectable

    def test_double_bit_detection(self):
        ecc = SECDEC_ECC()
        data = 0b1010
        encoded = ecc.encode(data)
        # Flip two bits
        corrupted = encoded ^ 0b11
        _, uncorrectable = ecc.decode(corrupted)
        assert uncorrectable

    def test_bitstream_roundtrip(self):
        ecc = SECDEC_ECC()
        bs = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        encoded = ecc.encode_bitstream(bs)
        decoded, n_unc = ecc.decode_bitstream(encoded)
        np.testing.assert_array_equal(decoded[: len(bs)], bs)
        assert n_unc == 0

    def test_bitstream_single_bit_correction(self):
        ecc = SECDEC_ECC()
        bs = np.array([1, 1, 0, 1, 0, 1, 0, 0], dtype=np.uint8)
        encoded = ecc.encode_bitstream(bs)
        encoded[5] ^= 1  # corrupt 1 bit
        decoded, n_unc = ecc.decode_bitstream(encoded)
        np.testing.assert_array_equal(decoded[: len(bs)], bs)
        assert n_unc == 0

    def test_bitstream_double_bit_detected(self):
        ecc = SECDEC_ECC()
        bs = np.array([1, 0, 1, 0], dtype=np.uint8)
        encoded = ecc.encode_bitstream(bs)
        encoded[0] ^= 1
        encoded[1] ^= 1
        _, n_unc = ecc.decode_bitstream(encoded)
        assert n_unc > 0

    def test_secded_8_bit_encoding(self):
        ecc = SECDEC_ECC()
        bs = np.array([1, 0, 1, 1], dtype=np.uint8)
        encoded = ecc.encode_bitstream(bs)
        assert len(encoded) == 8  # 4 data → 8 SECDED


# ── AdaptationPolicy Tests ──────────────────────────────────────────


class TestAdaptationPolicy:
    def test_high_scc_doubles_length(self):
        policy = AdaptationPolicy(scc_high=0.15)
        config = RuntimeConfig(bitstream_length=256)
        new, trigger = policy.decide(config, {"ema_scc": 0.20})
        assert trigger == "high_scc"
        assert new.bitstream_length == 512

    def test_low_scc_halves_length(self):
        policy = AdaptationPolicy(scc_low=0.05)
        config = RuntimeConfig(bitstream_length=512)
        new, trigger = policy.decide(config, {"ema_scc": 0.03})
        assert trigger == "low_scc"
        assert new.bitstream_length == 256

    def test_low_scc_floor_at_min(self):
        policy = AdaptationPolicy(scc_low=0.05, min_length=256)
        config = RuntimeConfig(bitstream_length=256)
        new, trigger = policy.decide(config, {"ema_scc": 0.03})
        assert trigger is None

    def test_high_scc_enables_ecc(self):
        policy = AdaptationPolicy(scc_high=0.10, ecc_trigger_length=2048)
        config = RuntimeConfig(bitstream_length=2048)
        new, trigger = policy.decide(config, {"ema_scc": 0.20})
        assert new.ecc_enabled is True

    def test_drift_switches_decorrelator(self):
        policy = AdaptationPolicy(enable_decorrelator_cascade=False)
        config = RuntimeConfig(decorrelator=DecorrelatorType.LFSR)
        new, trigger = policy.decide(config, {"ema_scc": 0.08, "drift_detected": True})
        assert trigger == "decorrelator_drift"
        assert new.decorrelator == DecorrelatorType.SOBOL

    def test_stable_no_adaptation(self):
        policy = AdaptationPolicy()
        config = RuntimeConfig(bitstream_length=512)
        new, trigger = policy.decide(config, {"ema_scc": 0.10, "drift_detected": False})
        assert trigger is None

    def test_decorrelator_cascade_lfsr_to_sobol(self):
        policy = AdaptationPolicy(enable_decorrelator_cascade=True)
        config = RuntimeConfig(decorrelator=DecorrelatorType.LFSR)
        new, trigger = policy.decide(config, {"ema_scc": 0.08, "drift_detected": True})
        assert trigger == "decorrelator_cascade"
        assert new.decorrelator == DecorrelatorType.SOBOL

    def test_decorrelator_cascade_sobol_to_halton(self):
        policy = AdaptationPolicy(enable_decorrelator_cascade=True)
        config = RuntimeConfig(decorrelator=DecorrelatorType.SOBOL)
        new, trigger = policy.decide(config, {"ema_scc": 0.08, "drift_detected": True})
        assert trigger == "decorrelator_cascade"
        assert new.decorrelator == DecorrelatorType.HALTON

    def test_decorrelator_cascade_halton_to_hybrid(self):
        policy = AdaptationPolicy(enable_decorrelator_cascade=True)
        config = RuntimeConfig(decorrelator=DecorrelatorType.HALTON)
        new, trigger = policy.decide(config, {"ema_scc": 0.08, "drift_detected": True})
        assert trigger == "decorrelator_cascade"
        assert new.decorrelator == DecorrelatorType.HYBRID

    def test_decorrelator_cascade_hybrid_stays(self):
        policy = AdaptationPolicy(enable_decorrelator_cascade=True)
        config = RuntimeConfig(decorrelator=DecorrelatorType.HYBRID)
        new, trigger = policy.decide(config, {"ema_scc": 0.08, "drift_detected": True})
        assert trigger is None  # already at top of cascade

    def test_next_decorrelator_off_cascade_returns_current(self, monkeypatch):
        # Guards against the cascade table and the DecorrelatorType enum drifting
        # out of sync: a decorrelator missing from the cascade is left unchanged
        # rather than raising. Simulate the drift by shrinking the cascade.
        import sc_neurocore.control.sc_runtime as sc_runtime_module

        monkeypatch.setattr(sc_runtime_module, "DECORRELATOR_CASCADE", [DecorrelatorType.LFSR])
        result = AdaptationPolicy._next_decorrelator(DecorrelatorType.HYBRID)
        assert result == DecorrelatorType.HYBRID


# ── RuntimeReport Tests ─────────────────────────────────────────────


class TestRuntimeReport:
    def test_adaptation_rate(self):
        report = RuntimeReport(total_observations=100)
        from sc_neurocore.control.sc_runtime import AdaptationEvent
        import time

        for _ in range(10):
            report.adaptations.append(
                AdaptationEvent(
                    timestamp_ns=time.perf_counter_ns(),
                    trigger="test",
                    old_config={},
                    new_config={},
                    metric_value=0.0,
                )
            )
        assert report.adaptation_rate() == pytest.approx(0.1)

    def test_adaptation_rate_zero(self):
        report = RuntimeReport(total_observations=0)
        assert report.adaptation_rate() == 0.0

    def test_adaptation_rate_last_n_window(self):
        from sc_neurocore.control.sc_runtime import AdaptationEvent

        report = RuntimeReport(total_observations=100)
        for _ in range(10):
            report.adaptations.append(
                AdaptationEvent(
                    timestamp_ns=0,
                    trigger="test",
                    old_config={},
                    new_config={},
                    metric_value=0.0,
                )
            )
        # last_n=5 windows the rate over the five most recent adaptations.
        assert report.adaptation_rate(last_n=5) == pytest.approx(1.0)

    def test_summary_includes_ecc_mode(self):
        config = RuntimeConfig(ecc_enabled=True, ecc_mode=ECCMode.SECDED)
        report = RuntimeReport(total_observations=5, final_config=config)
        s = report.summary()
        assert "secded_8_4" in s

    def test_summary_includes_uncorrectable(self):
        report = RuntimeReport(total_observations=1, uncorrectable_errors=3)
        s = report.summary()
        assert "Uncorrectable errors: 3" in s


# ── SCRuntimeEngine Tests ────────────────────────────────────────────


class TestSCRuntimeEngine:
    def test_observe_returns_metrics(self):
        engine = SCRuntimeEngine()
        bs = np.ones(100, dtype=np.uint8)
        result = engine.observe(bs)
        assert "density" in result
        assert "adapted" in result
        assert "config_ecc_mode" in result

    def test_adaptation_on_high_correlation(self):
        engine = SCRuntimeEngine(
            initial_config=RuntimeConfig(bitstream_length=256),
            policy=AdaptationPolicy(scc_high=0.10),
        )
        a = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        b = a.copy()
        for _ in range(20):
            r = engine.observe(a, reference=b)
        assert engine.report.num_adaptations > 0

    def test_protect_with_hamming(self):
        engine = SCRuntimeEngine(
            initial_config=RuntimeConfig(ecc_enabled=True, ecc_mode=ECCMode.HAMMING),
        )
        bs = np.array([1, 0, 1, 1], dtype=np.uint8)
        protected = engine.protect(bs)
        assert len(protected) > len(bs)

    def test_protect_with_secded(self):
        engine = SCRuntimeEngine(
            initial_config=RuntimeConfig(ecc_enabled=True, ecc_mode=ECCMode.SECDED),
        )
        bs = np.array([1, 0, 1, 1], dtype=np.uint8)
        protected = engine.protect(bs)
        assert len(protected) == 8  # 4 data → 8 SECDED

    def test_protect_with_parity(self):
        engine = SCRuntimeEngine(
            initial_config=RuntimeConfig(ecc_enabled=True, ecc_mode=ECCMode.PARITY),
        )
        bs = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        protected = engine.protect(bs)
        assert len(protected) == 9  # 8 data + 1 parity

    def test_protect_without_ecc(self):
        engine = SCRuntimeEngine(
            initial_config=RuntimeConfig(ecc_enabled=False),
        )
        bs = np.array([1, 0, 1, 1], dtype=np.uint8)
        protected = engine.protect(bs)
        np.testing.assert_array_equal(protected, bs)

    def test_protect_ecc_enabled_mode_none_passthrough(self):
        # ecc_enabled with ECCMode.NONE selects no concrete codec, so protect
        # returns the bitstream unchanged.
        engine = SCRuntimeEngine(
            initial_config=RuntimeConfig(ecc_enabled=True, ecc_mode=ECCMode.NONE),
        )
        bs = np.array([1, 0, 1, 1], dtype=np.uint8)
        np.testing.assert_array_equal(engine.protect(bs), bs)

    def test_recover_without_ecc_passthrough(self):
        engine = SCRuntimeEngine(
            initial_config=RuntimeConfig(ecc_enabled=False),
        )
        bs = np.array([1, 0, 1, 1], dtype=np.uint8)
        np.testing.assert_array_equal(engine.recover(bs), bs)

    def test_recover_ecc_enabled_mode_none_passthrough(self):
        # ecc_enabled with ECCMode.NONE selects no concrete codec, so recover
        # returns the encoded stream unchanged.
        engine = SCRuntimeEngine(
            initial_config=RuntimeConfig(ecc_enabled=True, ecc_mode=ECCMode.NONE),
        )
        bs = np.array([1, 0, 1, 1], dtype=np.uint8)
        np.testing.assert_array_equal(engine.recover(bs), bs)

    def test_protect_recover_roundtrip_hamming(self):
        engine = SCRuntimeEngine(
            initial_config=RuntimeConfig(ecc_enabled=True, ecc_mode=ECCMode.HAMMING),
        )
        bs = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        protected = engine.protect(bs)
        recovered = engine.recover(protected)
        np.testing.assert_array_equal(recovered[: len(bs)], bs)

    def test_protect_recover_roundtrip_secded(self):
        engine = SCRuntimeEngine(
            initial_config=RuntimeConfig(ecc_enabled=True, ecc_mode=ECCMode.SECDED),
        )
        bs = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        protected = engine.protect(bs)
        recovered = engine.recover(protected)
        np.testing.assert_array_equal(recovered[: len(bs)], bs)

    def test_protect_recover_roundtrip_parity(self):
        engine = SCRuntimeEngine(
            initial_config=RuntimeConfig(ecc_enabled=True, ecc_mode=ECCMode.PARITY),
        )
        bs = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        protected = engine.protect(bs)
        recovered = engine.recover(protected)
        np.testing.assert_array_equal(recovered[: len(bs)], bs)

    def test_secded_detects_double_errors(self):
        engine = SCRuntimeEngine(
            initial_config=RuntimeConfig(ecc_enabled=True, ecc_mode=ECCMode.SECDED),
        )
        bs = np.array([1, 0, 1, 0], dtype=np.uint8)
        protected = engine.protect(bs)
        protected[0] ^= 1
        protected[1] ^= 1
        engine.recover(protected)
        assert engine.report.uncorrectable_errors > 0

    def test_report_tracks_observations(self):
        engine = SCRuntimeEngine()
        for _ in range(5):
            engine.observe(np.ones(50, dtype=np.uint8))
        assert engine.report.total_observations == 5

    def test_report_summary(self):
        engine = SCRuntimeEngine()
        engine.observe(np.ones(50, dtype=np.uint8))
        s = engine.report.summary()
        assert "observations" in s

    def test_batch_protect_recover(self):
        engine = SCRuntimeEngine(
            initial_config=RuntimeConfig(ecc_enabled=True, ecc_mode=ECCMode.HAMMING),
        )
        batch = [
            np.array([1, 0, 1, 0], dtype=np.uint8),
            np.array([0, 1, 0, 1], dtype=np.uint8),
        ]
        protected = engine.protect_batch(batch)
        assert len(protected) == 2
        for p in protected:
            assert len(p) > 4
        recovered = engine.recover_batch(protected)
        for i, r in enumerate(recovered):
            np.testing.assert_array_equal(r[:4], batch[i])

    def test_observe_returns_activity_zone(self):
        engine = SCRuntimeEngine()
        r = engine.observe(np.zeros(100, dtype=np.uint8))
        assert r["activity_zone"] == "idle"

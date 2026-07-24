# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSCRuntimeEngine from former test_sc_runtime.py

"""Focused suite: TestSCRuntimeEngine from former test_sc_runtime.py."""

from __future__ import annotations

from sc_runtime_support import *  # noqa: F403


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

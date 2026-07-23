# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPythonFallbackFunctions from former test_predictive_spike_codec.py

"""Focused suite: TestPythonFallbackFunctions from former test_predictive_spike_codec.py."""

from __future__ import annotations

from tests.predictive_spike_codec_support import *  # noqa: F403

class TestPythonFallbackFunctions:
    """Direct tests for Python fallback functions.

    These are never hit when the Rust engine is available (CI builds Rust).
    Testing them directly validates Python fallback behaviour regardless of Rust availability.
    """

    def test_predict_and_xor_roundtrip(self) -> None:
        from sc_neurocore.spike_codec.predictive_codec import (
            _predict_and_xor,
            _xor_and_recover,
        )

        rng = np.random.RandomState(42)
        spikes = (rng.random((200, 16)) < 0.05).astype(np.int8)
        errors, correct = _predict_and_xor(spikes, 16, 0.005, 0.5)
        assert errors.shape == spikes.shape
        assert correct >= 0
        recovered = _xor_and_recover(errors, 16, 0.005, 0.5)
        np.testing.assert_array_equal(recovered, spikes)

    def test_predict_and_xor_silent(self) -> None:
        from sc_neurocore.spike_codec.predictive_codec import (
            _predict_and_xor,
            _xor_and_recover,
        )

        spikes = np.zeros((100, 8), dtype=np.int8)
        errors, correct = _predict_and_xor(spikes, 8, 0.01, 0.5)
        assert correct == 800  # all correct (predict 0, actual 0)
        recovered = _xor_and_recover(errors, 8, 0.01, 0.5)
        np.testing.assert_array_equal(recovered, spikes)

    def test_lfsr16_step(self) -> None:
        from sc_neurocore.spike_codec.predictive_codec import _lfsr16_step

        reg = 0xACE1
        seen = set()
        for _ in range(100):
            reg = _lfsr16_step(reg)
            assert 0 <= reg <= 0xFFFF
            seen.add(reg)
        assert len(seen) == 100  # no short cycle

    def test_predict_and_xor_lfsr_roundtrip(self) -> None:
        from sc_neurocore.spike_codec.predictive_codec import (
            _predict_and_xor_lfsr,
            _xor_and_recover_lfsr,
        )

        rng = np.random.RandomState(42)
        spikes = (rng.random((200, 16)) < 0.05).astype(np.int8)
        errors, correct = _predict_and_xor_lfsr(spikes, 16, 1, 0xACE1)
        assert errors.shape == spikes.shape
        assert correct >= 0
        recovered = _xor_and_recover_lfsr(errors, 16, 1, 0xACE1)
        np.testing.assert_array_equal(recovered, spikes)

    def test_predict_and_xor_lfsr_all_firing(self) -> None:
        from sc_neurocore.spike_codec.predictive_codec import (
            _predict_and_xor_lfsr,
            _xor_and_recover_lfsr,
        )

        spikes = np.ones((50, 4), dtype=np.int8)
        errors, _ = _predict_and_xor_lfsr(spikes, 4, 50, 0xBEEF)
        recovered = _xor_and_recover_lfsr(errors, 4, 50, 0xBEEF)
        np.testing.assert_array_equal(recovered, spikes)

    def test_legacy_rate_predictor_contract(self) -> None:
        """Legacy EMA predictor should update, predict, and reset deterministically."""
        from sc_neurocore.spike_codec.predictive_codec import _RatePredictor

        predictor = _RatePredictor(n_channels=3, alpha=0.5, threshold=0.25)
        np.testing.assert_array_equal(predictor.predict(), np.zeros(3, dtype=np.int8))

        predictor.update(np.array([1, 0, 1], dtype=np.int8))

        np.testing.assert_allclose(predictor.rates, np.array([0.5, 0.0, 0.5]))
        np.testing.assert_array_equal(predictor.predict(), np.array([1, 0, 1], dtype=np.int8))

        predictor.reset()

        np.testing.assert_allclose(predictor.rates, np.zeros(3, dtype=np.float64))

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPythonFallbackPath from former test_predictive_spike_codec.py

"""Focused suite: TestPythonFallbackPath from former test_predictive_spike_codec.py."""

from __future__ import annotations

from tests.predictive_spike_codec_support import *  # noqa: F403

class TestPythonFallbackPath:
    """Force Python path through the class by monkeypatching _HAS_RUST."""

    def test_ema_python_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sc_neurocore.spike_codec.predictive_codec as mod

        monkeypatch.setattr(mod, "_HAS_RUST", False)
        rng = np.random.RandomState(42)
        spikes = (rng.random((200, 16)) < 0.05).astype(np.int8)
        codec = PredictiveSpikeCodec(predictor="ema")
        data, result = codec.compress(spikes)
        recovered = codec.decompress(data, 200, 16)
        np.testing.assert_array_equal(recovered, spikes)
        assert result.predictor_type == "ema"

    def test_lfsr_python_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sc_neurocore.spike_codec.predictive_codec as mod

        monkeypatch.setattr(mod, "_HAS_RUST", False)
        rng = np.random.RandomState(42)
        spikes = (rng.random((200, 16)) < 0.05).astype(np.int8)
        codec = PredictiveSpikeCodec(predictor="lfsr", alpha_q8=1, seed=0xACE1)
        data, result = codec.compress(spikes)
        recovered = codec.decompress(data, 200, 16)
        np.testing.assert_array_equal(recovered, spikes)
        assert result.predictor_type == "lfsr"

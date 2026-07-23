# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBCIEncoder from former test_bci.py

"""Focused suite: TestBCIEncoder from former test_bci.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent))
from bci_support import *  # noqa: F403

class TestBCIEncoder:
    def test_encode_shape(self):
        enc = BCIEncoder(n_channels=4, seed=42)
        spikes = enc.encode(np.array([0.1, 0.5, 0.8, 0.3]), T=20)
        assert spikes.shape == (20, 4)
        assert spikes.dtype == np.int8

    def test_encode_binary_output(self):
        enc = BCIEncoder(n_channels=8, seed=42)
        spikes = enc.encode(np.random.randn(8), T=50)
        assert set(np.unique(spikes).tolist()) <= {0, 1}

    def test_encode_deterministic(self):
        """Same seed → same output, always."""
        enc1 = BCIEncoder(n_channels=4, seed=99)
        enc2 = BCIEncoder(n_channels=4, seed=99)
        signal = np.array([0.2, 0.4, 0.6, 0.8])
        assert np.array_equal(enc1.encode(signal, T=30), enc2.encode(signal, T=30))

    def test_encode_2d_input(self):
        """Multi-sample input averaged per channel."""
        enc = BCIEncoder(n_channels=3, seed=42)
        signal = np.random.randn(3, 100)
        spikes = enc.encode(signal, T=20)
        assert spikes.shape == (20, 3)

    def test_encode_stream(self):
        enc = BCIEncoder(n_channels=4, sampling_rate=20000, window_ms=1.0, seed=42)
        signal = np.random.RandomState(42).randn(4, 1000)
        stream = enc.encode_stream(signal)
        assert stream.shape[1] == 4
        assert stream.shape[0] > 0
        assert stream.dtype == np.int8

    def test_encode_stream_empty(self):
        enc = BCIEncoder(n_channels=2, sampling_rate=20000, window_ms=1.0)
        signal = np.random.randn(2, 5)  # too short for one window
        stream = enc.encode_stream(signal)
        assert stream.shape[1] == 2

    def test_normalize_flat_signal(self):
        result = BCIEncoder._normalize(np.array([5.0, 5.0, 5.0]))
        assert np.allclose(result, 0.5)

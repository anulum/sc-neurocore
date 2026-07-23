# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTraceEncoder from former test_identity_substrate.py

"""Focused suite: TestTraceEncoder from former test_identity_substrate.py."""

from __future__ import annotations

from tests.identity_substrate_support import *  # noqa: F403

class TestTraceEncoder:
    def test_encode_shape(self):
        enc = TraceEncoder(n_neurons=N_CORTICAL, hash_dims=16, seed=42)
        pattern = enc.encode("The cat sat on the mat.", duration_ms=50, dt=0.001)
        assert pattern.shape == (N_CORTICAL, 50)

    def test_encode_produces_spikes(self):
        enc = TraceEncoder(n_neurons=N_CORTICAL, hash_dims=16, seed=42)
        pattern = enc.encode("Reasoning about identity and memory.", duration_ms=100, dt=0.001)
        assert pattern.sum() > 0

    def test_encode_empty_text(self):
        enc = TraceEncoder(n_neurons=N_CORTICAL, hash_dims=16, seed=42)
        pattern = enc.encode("", duration_ms=50, dt=0.001)
        assert pattern.shape == (N_CORTICAL, 50)

    def test_encode_key_value(self):
        enc = TraceEncoder(n_neurons=N_CORTICAL, hash_dims=16, seed=42)
        pattern = enc.encode_key_value("decision", "use PCA for dimensionality reduction")
        assert pattern.shape[0] == N_CORTICAL
        assert pattern.shape[1] > 0

    def test_different_texts_produce_different_patterns(self):
        enc = TraceEncoder(n_neurons=N_CORTICAL, hash_dims=16, seed=42)
        p1 = enc.encode("Alpha bravo charlie.", duration_ms=50, dt=0.001)
        p2 = enc.encode("Delta echo foxtrot.", duration_ms=50, dt=0.001)
        assert not np.array_equal(p1, p2)

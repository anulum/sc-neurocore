# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPOYODecoder from former test_neural_decoders.py

"""Focused suite: TestPOYODecoder from former test_neural_decoders.py."""

from __future__ import annotations

from tests.neural_decoders_support import *  # noqa: F403

class TestPOYODecoder:
    def test_defaults(self) -> None:
        dec = POYODecoder()
        assert dec.d_model == 64
        assert dec.n_latents == 32

    def test_encode_empty(self) -> None:
        dec = POYODecoder()
        latents = dec.encode([])
        assert latents.shape == (32, 64)
        assert np.allclose(latents, 0.0)

    def test_encode_shape(self) -> None:
        dec = POYODecoder(d_model=16, n_latents=8)
        trains = [np.zeros(100) for _ in range(5)]
        for i, t in enumerate(trains):
            t[i * 10 + 5] = 1
        latents = dec.encode(trains)
        assert latents.shape == (8, 16)

    def test_different_activity_different_latents(self) -> None:
        dec = POYODecoder(d_model=16, n_latents=4, seed=7)
        t1 = [np.zeros(100)]
        t1[0][10] = 1
        t2 = [np.zeros(100)]
        t2[0][90] = 1
        l1 = dec.encode(t1)
        l2 = dec.encode(t2)
        assert not np.allclose(l1, l2)

    def test_decode_shape(self) -> None:
        dec = POYODecoder(d_model=16, n_latents=8)
        latents = np.random.default_rng(1).normal(0, 1, (8, 16))
        queries = np.random.default_rng(2).normal(0, 1, (3, 16))
        out = dec.decode(latents, queries)
        assert out.shape == (3, 16)

    def test_reset_clears_embeddings(self) -> None:
        dec = POYODecoder(d_model=8, n_latents=4)
        train = np.zeros(10)
        train[5] = 1
        dec.encode([train])
        assert len(dec._unit_embeddings) > 0
        dec.reset()
        assert len(dec._unit_embeddings) == 0

    def test_deterministic(self) -> None:
        dec = POYODecoder(d_model=16, n_latents=4, seed=42)
        trains = [np.zeros(50)]
        trains[0][10] = 1
        trains[0][30] = 1
        l1 = dec.encode(trains).copy()
        dec.reset()
        l2 = dec.encode(trains)
        np.testing.assert_array_equal(l1, l2)

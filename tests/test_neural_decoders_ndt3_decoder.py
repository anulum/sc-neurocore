# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNDT3Decoder from former test_neural_decoders.py

"""Focused suite: TestNDT3Decoder from former test_neural_decoders.py."""

from __future__ import annotations

from tests.neural_decoders_support import *  # noqa: F403

class TestNDT3Decoder:
    def test_defaults(self) -> None:
        dec = NDT3Decoder()
        assert dec.d_model == 64
        assert dec.bin_size_ms == pytest.approx(20.0)

    def test_bin_and_embed_empty(self) -> None:
        dec = NDT3Decoder(d_model=8)
        binned, embedded = dec.bin_and_embed([])
        assert binned.shape[0] == 0

    def test_bin_and_embed_shape(self) -> None:
        dec = NDT3Decoder(d_model=16, bin_size_ms=10.0)
        trains = [np.zeros(100), np.zeros(100)]  # 100 steps @ dt=1 → 10 bins
        trains[0][5] = 1
        binned, embedded = dec.bin_and_embed(trains, dt=1.0)
        assert binned.shape == (10, 2)
        assert embedded.shape == (10, 16)

    def test_binning_counts_spikes(self) -> None:
        dec = NDT3Decoder(bin_size_ms=10.0)
        train = np.zeros(100)
        train[3] = 1
        train[7] = 1
        train[15] = 1
        binned, _ = dec.bin_and_embed([train], dt=1.0)
        assert binned[0, 0] == pytest.approx(2.0)  # bin 0: indices 0-9
        assert binned[1, 0] == pytest.approx(1.0)  # bin 1: indices 10-19

    def test_causal_mask_in_predict(self) -> None:
        """First bin prediction depends only on itself (causal)."""
        dec = NDT3Decoder(d_model=8, bin_size_ms=5.0, seed=42)
        trains = [np.zeros(50)]
        trains[0][2] = 1
        trains[0][42] = 1
        _, emb1 = dec.bin_and_embed(trains, dt=1.0)
        out = dec.predict_next(emb1)
        # Modify later bins, first bin output should not change
        trains2 = [np.zeros(50)]
        trains2[0][2] = 1
        trains2[0][45] = 1  # different late spike
        _, emb2 = dec.bin_and_embed(trains2, dt=1.0)
        out2 = dec.predict_next(emb2)
        np.testing.assert_allclose(out[0], out2[0], atol=1e-10)

    def test_decode_pipeline(self) -> None:
        dec = NDT3Decoder(d_model=8, bin_size_ms=5.0)
        trains = [np.zeros(30) for _ in range(3)]
        for i, t in enumerate(trains):
            t[i * 5 + 2] = 1
        out = dec.decode(trains)
        assert out.shape[1] == 8
        assert out.shape[0] > 0

    def test_different_activity_different_output(self) -> None:
        dec = NDT3Decoder(d_model=8, bin_size_ms=10.0, seed=99)
        t1 = [np.zeros(50)]
        t1[0][5] = 1
        t2 = [np.zeros(50)]
        t2[0][45] = 1
        o1 = dec.decode(t1)
        o2 = dec.decode(t2)
        assert not np.allclose(o1, o2)

    def test_bin_and_embed_train_shorter_than_one_bin(self) -> None:
        """Trains too short to fill a single 20 ms bin yield no bins at all,
        keeping the neuron dimension on the (empty) binned matrix."""
        dec = NDT3Decoder(d_model=8)
        trains = [np.zeros(5), np.zeros(5)]
        binned, embedded = dec.bin_and_embed(trains, dt=1.0)
        assert binned.shape == (0, 2)
        assert embedded.shape == (0, 8)

    def test_predict_next_on_empty_embedding(self) -> None:
        """Predicting from an empty embedding short-circuits to an empty
        output rather than running attention over zero positions."""
        dec = NDT3Decoder(d_model=8)
        empty = np.zeros((0, 8))
        out = dec.predict_next(empty)
        assert out.shape == (0, 8)

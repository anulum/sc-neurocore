# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestConvertedSNN from former test_conversion_ann_snn.py

"""Focused suite: TestConvertedSNN from former test_conversion_ann_snn.py."""

from __future__ import annotations

from tests.conversion_ann_snn_support import *  # noqa: F403

class TestConvertedSNN:
    def _make_snn(self) -> ConvertedSNN:
        rng = np.random.default_rng(11)
        weights: list[npt.NDArray[np.float64]] = [
            rng.normal(size=(10, 5)) * 0.1,
            rng.normal(size=(3, 10)) * 0.1,
        ]
        biases: list[npt.NDArray[np.float64] | None] = [np.zeros(10), np.zeros(3)]
        return ConvertedSNN(
            weights=weights,
            biases=biases,
            thresholds=[1.0, 1.0],
            T=16,
        )

    def test_n_layers(self) -> None:
        snn = self._make_snn()
        assert snn.n_layers == 2

    def test_run_single(self) -> None:
        snn = self._make_snn()
        x = np.random.default_rng(12).random(5)
        out = snn.run(x)
        assert out.shape == (3,)
        assert out.dtype == np.float64

    def test_run_batch(self) -> None:
        snn = self._make_snn()
        x = np.random.default_rng(13).random((4, 5))
        out = snn.run(x)
        assert out.shape == (4, 3)

    def test_classify(self) -> None:
        snn = self._make_snn()
        x = np.random.default_rng(14).random((4, 5))
        preds = snn.classify(x)
        assert preds.shape == (4,)
        assert all(0 <= p < 3 for p in preds)

    def test_classify_single(self) -> None:
        snn = self._make_snn()
        x = np.random.default_rng(15).random(5)
        pred = snn.classify(x)
        assert isinstance(int(pred), int)

    def test_no_bias(self) -> None:
        weights: list[npt.NDArray[np.float64]] = [np.random.default_rng(16).normal(size=(3, 2))]
        biases: list[npt.NDArray[np.float64] | None] = [None]
        snn = ConvertedSNN(weights=weights, biases=biases, thresholds=[1.0], T=8)
        out = snn.run(np.random.default_rng(17).random(2))
        assert out.shape == (3,)

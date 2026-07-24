# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeContrastiveLoss from former test_contrastive.py

"""Focused suite: TestSpikeContrastiveLoss from former test_contrastive.py."""

from __future__ import annotations

from tests.contrastive_support import *  # noqa: F403


class TestSpikeContrastiveLoss:
    def test_compute_matches_manual_infonce(self) -> None:
        loss_fn = SpikeContrastiveLoss(temperature=0.5)
        view_a = np.array(
            [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.4, 0.2, 1.0]],
            dtype=np.float64,
        )
        view_b = np.array(
            [[0.9, 0.1, 0.5], [0.1, 0.9, 0.5], [0.3, 0.3, 1.0]],
            dtype=np.float64,
        )

        a_norm = view_a / np.linalg.norm(view_a, axis=1, keepdims=True)
        b_norm = view_b / np.linalg.norm(view_b, axis=1, keepdims=True)
        logits = a_norm @ b_norm.T / 0.5
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        expected = -float(np.log(np.diag(exp_logits) / exp_logits.sum(axis=1)).mean())

        assert loss_fn.compute(view_a, view_b) == pytest.approx(expected)

    def test_identical_views_have_lower_loss_than_permuted_views(self) -> None:
        view = np.eye(4, dtype=np.float64)
        permuted = view[[1, 2, 3, 0]]

        loss_fn = SpikeContrastiveLoss(temperature=0.25)

        assert loss_fn.compute(view, view) < loss_fn.compute(view, permuted)

    def test_single_sample_has_no_negatives(self) -> None:
        assert SpikeContrastiveLoss().compute(np.random.rand(1, 4), np.random.rand(1, 4)) == 0.0

    @pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf")])
    def test_temperature_must_be_finite_and_positive(self, temperature: float) -> None:
        with pytest.raises(ValueError, match="temperature"):
            SpikeContrastiveLoss(temperature=temperature)

    @pytest.mark.parametrize(
        ("view_a", "view_b", "message"),
        [
            (np.ones(4), np.ones((2, 2)), "2-D"),
            (np.ones((2, 2)), np.ones((3, 2)), "same shape"),
            (np.ones((2, 0)), np.ones((2, 0)), "at least one feature"),
            (np.array([[1.0, np.nan], [0.0, 1.0]]), np.ones((2, 2)), "finite"),
        ],
    )
    def test_compute_rejects_invalid_views(
        self,
        view_a: FloatArray,
        view_b: FloatArray,
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            SpikeContrastiveLoss().compute(view_a, view_b)

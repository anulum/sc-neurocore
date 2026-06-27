# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import pytest

from sc_neurocore.contrastive import CSDPRule, SpikeContrastiveLoss

FloatArray = NDArray[np.float64]


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


class TestCSDPRule:
    def test_positive_update_matches_hebbian_rule(self) -> None:
        weights = np.array([[0.2, 0.4, 0.6], [0.1, 0.3, 0.5]], dtype=np.float64)
        pre = np.array([1.0, 0.0, 0.5], dtype=np.float64)
        post = np.array([0.25, 0.75], dtype=np.float64)
        rule = CSDPRule(lr=0.2, decay=0.05)

        expected = weights + 0.2 * np.outer(post, pre) - 0.05 * weights

        np.testing.assert_allclose(rule.positive_update(weights, pre, post), expected)

    def test_negative_update_matches_anti_hebbian_rule(self) -> None:
        weights = np.array([[0.2, 0.4, 0.6], [0.1, 0.3, 0.5]], dtype=np.float64)
        pre = np.array([0.5, 0.0, 1.0], dtype=np.float64)
        post = np.array([0.25, 0.75], dtype=np.float64)
        rule = CSDPRule(lr=0.2, decay=0.05)

        expected = weights - 0.2 * np.outer(post, pre)

        np.testing.assert_allclose(rule.negative_update(weights, pre, post), expected)

    def test_contrastive_step_applies_positive_then_negative_phase(self) -> None:
        rule = CSDPRule(lr=0.1, decay=0.01)
        weights = np.array([[0.2, 0.4], [0.1, 0.3]], dtype=np.float64)
        pos_pre = np.array([1.0, 0.5], dtype=np.float64)
        pos_post = np.array([0.25, 1.0], dtype=np.float64)
        neg_pre = np.array([0.0, 1.0], dtype=np.float64)
        neg_post = np.array([0.5, 0.5], dtype=np.float64)

        after_positive = weights + 0.1 * np.outer(pos_post, pos_pre) - 0.01 * weights
        expected = after_positive - 0.1 * np.outer(neg_post, neg_pre)

        np.testing.assert_allclose(
            rule.contrastive_step(weights, pos_pre, pos_post, neg_pre, neg_post),
            expected,
        )

    def test_goodness_returns_sum_of_squares(self) -> None:
        rule = CSDPRule()
        assert rule.goodness(np.array([1.0, -2.0, 0.5])) == pytest.approx(5.25)

    def test_goodness_rejects_non_finite_activations(self) -> None:
        with pytest.raises(ValueError, match="activations"):
            CSDPRule().goodness(np.array([1.0, np.nan]))

    @pytest.mark.parametrize("field", ["lr", "decay"])
    def test_rule_parameters_must_be_finite_and_non_negative(self, field: str) -> None:
        with pytest.raises(ValueError, match=field):
            CSDPRule(**{field: float("nan")})
        with pytest.raises(ValueError, match=field):
            CSDPRule(**{field: -0.1})

    @pytest.mark.parametrize(
        ("pre", "post", "message"),
        [
            (np.ones((2, 2)), np.ones(2), "1-D"),
            (np.ones(3), np.ones((2, 1)), "1-D"),
            (np.array([1.0, np.inf]), np.ones(2), "finite"),
        ],
    )
    def test_updates_reject_invalid_spike_vectors(
        self,
        pre: FloatArray,
        post: FloatArray,
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            CSDPRule().positive_update(np.ones((2, 2)), pre, post)

    @pytest.mark.parametrize(
        ("weights", "message"),
        [
            (np.ones(4), "2-D"),
            (np.array([[1.0, np.nan], [0.0, 1.0]]), "finite"),
            (np.ones((3, 2)), "shape"),
        ],
    )
    def test_updates_reject_invalid_weight_matrices(
        self,
        weights: FloatArray,
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            CSDPRule().negative_update(weights, np.ones(2), np.ones(2))

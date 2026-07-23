# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCSDPRule from former test_contrastive.py

"""Focused suite: TestCSDPRule from former test_contrastive.py."""

from __future__ import annotations

from tests.contrastive_support import *  # noqa: F403

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

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPredictiveCodingEdgeBranches from former test_predictive_coding.py

"""Focused suite: TestPredictiveCodingEdgeBranches from former test_predictive_coding.py."""

from __future__ import annotations

from predictive_coding_support import *  # noqa: F403

class TestPredictiveCodingEdgeBranches:
    """Cover the no-op permutation, the empty-collection guards, and the
    error-history accessors before any prediction error is recorded."""

    def test_permute_by_multiple_of_length_returns_identity_copy(self) -> None:
        """A rotation that is a whole multiple of the dimension is the identity,
        so it returns an independent copy rather than rolling the bits."""
        hv = Hypervector.random(0x1234)
        rotated = hv.permute(0)
        assert np.array_equal(rotated.data, hv.data)
        # The result is a fresh array, not an alias of the source buffer.
        assert rotated.data is not hv.data

    def test_threshold_bundle_rejects_empty_input(self) -> None:
        """A majority vote over no vectors is undefined."""
        with pytest.raises(ValueError, match="cannot bundle zero vectors"):
            Hypervector.threshold_bundle([])

    def test_encode_sequence_rejects_empty_sequence(self) -> None:
        """An empty symbol sequence has no hypervector encoding."""
        encoder = SymbolEncoder(base_seed=7)
        with pytest.raises(ValueError, match="cannot encode empty sequence"):
            encoder.encode_sequence([])

    def test_error_accessors_on_fresh_layer(self) -> None:
        """Before any error is recorded, the recent-error mean is zero and the
        layer is not yet considered converged."""
        layer = PredictiveCodingLayer(input_dim=4, hidden_dim=3, seed=1)
        assert layer.mean_recent_error == 0.0
        assert layer.converged is False

    def test_mean_recent_error_after_a_few_updates(self) -> None:
        """With a populated but short history the mean is the recorded error and
        convergence still reports False (fewer than ten samples)."""
        layer = PredictiveCodingLayer(input_dim=4, hidden_dim=3, seed=1)
        observation = np.ones(4, dtype=np.float32)
        for _ in range(3):
            layer.compute_error(observation)
        # mu starts at zero so the prediction is zero and each error is unity.
        assert layer.mean_recent_error == pytest.approx(1.0)
        assert layer.converged is False

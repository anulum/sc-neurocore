# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Tests for stochastic-computing correlation regularizers

"""Tests for stochastic-computing correlation regularizers."""

import pytest

torch = pytest.importorskip("torch")

from sc_neurocore.training.sc_correlation_regularizers import (
    correlation_matrix,
    correlation_penalty,
    pairwise_correlation_penalty,
)


def test_independent_balanced_streams_have_zero_pairwise_penalty():
    streams = torch.tensor(
        [
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )

    assert pairwise_correlation_penalty(streams, threshold=0.05).item() == pytest.approx(0.0)


def test_identical_streams_are_penalized_above_threshold():
    streams = torch.tensor(
        [
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
        ]
    )

    assert pairwise_correlation_penalty(streams, threshold=0.2).item() > 0.6


def test_correlation_matrix_is_symmetric_with_unit_diagonal():
    streams = torch.tensor(
        [
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )

    corr = correlation_matrix(streams)

    assert torch.allclose(corr, corr.T)
    assert torch.allclose(torch.diagonal(corr), torch.ones(3))


def test_targeted_correlation_penalty_is_differentiable():
    observed = torch.tensor([0.1, 0.5, -0.2], requires_grad=True)
    penalty = correlation_penalty(observed, target=0.0, weight=2.0)

    penalty.backward()

    assert penalty.item() > 0.0
    assert observed.grad is not None
    assert observed.grad.abs().sum().item() > 0.0


def test_correlation_regularizers_reject_invalid_inputs():
    with pytest.raises(ValueError, match="streams"):
        pairwise_correlation_penalty(torch.ones(4), threshold=0.1)

    with pytest.raises(ValueError, match="threshold"):
        pairwise_correlation_penalty(torch.ones(2, 4), threshold=-0.1)

    with pytest.raises(ValueError, match="weight"):
        correlation_penalty(torch.ones(2), target=0.0, weight=-1.0)

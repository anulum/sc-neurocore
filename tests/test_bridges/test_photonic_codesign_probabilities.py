# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Photonic co-design adjacency probability contracts

"""Adjacency-to-probability contracts for stochastic photonic co-design."""

import numpy as np
import pytest

from sc_neurocore.bridges import derive_probabilities_from_adjacency


def test_derive_probabilities_from_adjacency_uses_inbound_weight_mass() -> None:
    adjacency = np.array(
        [
            [0.0, 2.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    probabilities = derive_probabilities_from_adjacency(adjacency)

    np.testing.assert_allclose(
        probabilities,
        np.array([1.0 / 65535.0, 1.0 - 1.0 / 65535.0, 1.0 / 3.0]),
    )


def test_derive_probabilities_rejects_non_square_and_empty() -> None:
    with pytest.raises(ValueError, match="square matrix"):
        derive_probabilities_from_adjacency(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="at least one node"):
        derive_probabilities_from_adjacency(np.zeros((0, 0)))


def test_derive_probabilities_zero_mass_falls_back_to_uniform_half() -> None:
    probs = derive_probabilities_from_adjacency(np.zeros((3, 3)))
    np.testing.assert_allclose(probs, np.full(3, 0.5))

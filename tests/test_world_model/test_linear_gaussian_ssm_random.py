# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — random Linear Gaussian state-space model contracts

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.world_model.predictive_model import LinearGaussianSSM


@pytest.mark.parametrize(
    ("state_dim", "obs_dim", "control_dim", "message"),
    [
        (0, 1, 0, "state_dim must be positive"),
        (1, 0, 0, "obs_dim must be positive"),
        (1, 1, -1, "control_dim must be non-negative"),
        (True, 1, 0, "state_dim must be an integer"),
    ],
)
def test_random_rejects_invalid_dimensions(
    state_dim: int, obs_dim: int, control_dim: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        LinearGaussianSSM.random(state_dim, obs_dim, control_dim)


def test_random_model_is_reproducible_and_stable() -> None:
    first = LinearGaussianSSM.random(4, 3, 2, seed=841)
    second = LinearGaussianSSM.random(4, 3, 2, seed=841)

    np.testing.assert_array_equal(first.A, second.A)
    assert float(np.max(np.abs(np.linalg.eigvals(first.A)))) < 1.0
    assert first.B.shape == (4, 2)
    assert first.D.shape == (3, 2)

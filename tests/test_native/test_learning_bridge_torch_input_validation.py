# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Torch learning input validation contracts

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sc_neurocore._native.learning_torch_support import validate_input


def test_validate_input_moves_dtype_and_checks_shape() -> None:
    values = torch.tensor([0, 1, 0], dtype=torch.int64)
    result = validate_input(
        values,
        name="spikes",
        count=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
        probability=True,
    )
    assert result.dtype == torch.float32
    with pytest.raises(ValueError, match="shape"):
        validate_input(
            values.reshape(1, 3),
            name="spikes",
            count=3,
            device=torch.device("cpu"),
            dtype=torch.float32,
            probability=True,
        )


@pytest.mark.parametrize("values", [[0.0, float("nan")], [-0.1, 0.0], [0.0, 1.1]])
def test_validate_input_rejects_unsafe_values(values: list[float]) -> None:
    with pytest.raises(ValueError):
        validate_input(
            torch.tensor(values),
            name="spikes",
            count=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
            probability=True,
        )


def test_validate_input_allows_unbounded_finite_rewards() -> None:
    result = validate_input(
        torch.tensor([-2.0, 3.0]),
        name="rewards",
        count=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        probability=False,
    )
    assert result.tolist() == [-2.0, 3.0]

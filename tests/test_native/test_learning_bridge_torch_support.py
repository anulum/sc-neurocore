# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Torch learning support tests

"""Exercise the vectorized validation path for larger Torch tensors."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sc_neurocore._native.learning_torch_support import validate_input


def _validate(values: object, *, probability: bool) -> object:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    return validate_input(
        tensor,
        name="values",
        count=65,
        device=torch.device("cpu"),
        dtype=torch.float32,
        probability=probability,
    )


def test_large_tensor_validation_accepts_finite_domains() -> None:
    """The vectorized path accepts probabilities and unrestricted rewards."""
    probability = _validate(torch.linspace(0.0, 1.0, 65), probability=True)
    reward = _validate(torch.linspace(-2.0, 2.0, 65), probability=False)
    assert probability.shape == reward.shape == (65,)


def test_large_tensor_validation_rejects_nonfinite_values() -> None:
    """A non-finite value fails before probability comparisons."""
    values = torch.zeros(65)
    values[32] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        _validate(values, probability=False)


@pytest.mark.parametrize(("index", "value"), [(0, -0.1), (64, 1.1)])
def test_large_tensor_validation_rejects_probability_bounds(index: int, value: float) -> None:
    """Both lower and upper probability violations fail closed."""
    values = torch.zeros(65)
    values[index] = value
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _validate(values, probability=True)

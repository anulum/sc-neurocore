# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Torch learning precision contracts

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeAlias, cast

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import Tensor

from sc_neurocore._native.learning_torch_precision import (
    normalise_bit_spec,
    normalise_clip,
    quantise_tensor,
)

BitSpec: TypeAlias = int | Sequence[int] | np.ndarray[Any, Any] | Tensor | None


@pytest.mark.parametrize(
    "spec",
    [
        4,
        [2, 3, 4],
        np.array([2, 3, 4], dtype=np.int32),
        np.array([2.0, 3.0, 4.0]),
        torch.tensor([2, 3, 4]),
        torch.tensor([2.0, 3.0, 4.0]),
    ],
)
def test_bit_specs_accept_integral_scalar_and_vectors(spec: object) -> None:
    result = normalise_bit_spec(
        cast(BitSpec, spec), count=3, device=torch.device("cpu"), field="weight_bits"
    )
    assert result is not None
    assert result.dtype == torch.int64 and result.numel() == 3


def test_bit_spec_none_disables_quantisation() -> None:
    assert normalise_bit_spec(None, count=3, device=torch.device("cpu"), field="bits") is None


@pytest.mark.parametrize(
    "spec",
    [
        True,
        [2, True, 4],
        np.array([True, False]),
        np.array(["2", "3"]),
        np.array([2.5, 3.0]),
        np.array([np.nan, 3.0]),
        torch.tensor([True, False]),
        torch.tensor([2.5, 3.0]),
        torch.tensor([float("inf"), 3.0]),
    ],
)
def test_bit_specs_reject_non_integral_values(spec: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        normalise_bit_spec(cast(BitSpec, spec), count=2, device=torch.device("cpu"), field="bits")


@pytest.mark.parametrize("spec", [[2], [2, 3], 1, 32])
def test_bit_specs_enforce_shape_and_bounds(spec: object) -> None:
    expected_error = ValueError
    if spec == [2]:
        result = normalise_bit_spec(
            cast(BitSpec, spec), count=3, device=torch.device("cpu"), field="bits"
        )
        assert result is not None and result.tolist() == [2, 2, 2]
        return
    with pytest.raises(expected_error):
        normalise_bit_spec(cast(BitSpec, spec), count=3, device=torch.device("cpu"), field="bits")


def test_clip_and_quantisation_helpers() -> None:
    assert normalise_clip(1, field="clip") == 1.0
    for value in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            normalise_clip(value, field="clip")
    values = torch.tensor([-2.0, 0.37, 2.0])
    assert quantise_tensor(values, None, 1.0) is values
    bits = torch.tensor([3, 3, 3])
    quantised = quantise_tensor(values, bits, 1.0)
    assert torch.all(quantised >= -1.0) and torch.all(quantised <= 1.0)

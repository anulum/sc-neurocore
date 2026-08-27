# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — stochastic-inference engine-binding contracts

"""Installed-extension contracts for packed stochastic inference."""

from __future__ import annotations

import importlib

import numpy as np
import numpy.typing as npt
import pytest

from tests.engine_requirement import require_engine

require_engine()
import sc_neurocore_engine as engine
from sc_neurocore.accel import sc_forward

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def _direct_inputs() -> tuple[
    npt.NDArray[np.uint64],
    int,
    int,
    int,
    npt.NDArray[np.float64],
    int,
    int,
]:
    return (
        np.asarray([np.iinfo(np.uint64).max], dtype=np.uint64),
        1,
        1,
        1,
        np.asarray([0.5], dtype=np.float64),
        64,
        0xACE1,
    )


def test_exported_name_signature_and_top_level_identity_are_stable() -> None:
    function = extension.py_sc_forward_packed

    assert function.__name__ == "py_sc_forward_packed"
    assert function.__module__ == "sc_neurocore_engine.sc_neurocore_engine"
    assert function.__text_signature__ == (
        "(weights_packed, n_out, n_in, n_words, input_probs, length, seed)"
    )
    assert engine.py_sc_forward_packed is function


def test_direct_binding_preserves_deterministic_value_shape_and_dtype() -> None:
    result = extension.py_sc_forward_packed(*_direct_inputs())

    np.testing.assert_array_equal(result, np.asarray([0.46875], dtype=np.float64))
    assert result.shape == (1,)
    assert result.dtype == np.float64


def test_production_rust_dispatch_is_bit_exact_with_direct_binding() -> None:
    weights, _, _, _, probs, length, seed = _direct_inputs()
    packed = weights.reshape(1, 1, 1)

    direct = extension.py_sc_forward_packed(*_direct_inputs())
    public = sc_forward(packed, probs, length=length, backend="rust", seed=seed)

    np.testing.assert_array_equal(public, direct)


@pytest.mark.parametrize(
    ("weights", "probs", "length", "message"),
    (
        (
            np.asarray([0], dtype=np.uint64),
            np.asarray([0.0], dtype=np.float64),
            0,
            "length must be positive, got 0",
        ),
        (
            np.asarray([0, 0], dtype=np.uint64),
            np.asarray([0.0], dtype=np.float64),
            64,
            "weights_packed length must be n_out*n_in*n_words (1), got 2",
        ),
        (
            np.asarray([0], dtype=np.uint64),
            np.asarray([0.0, 0.0], dtype=np.float64),
            64,
            "input_probs length must be n_in (1), got 2",
        ),
        (
            np.asarray([0], dtype=np.uint64),
            np.asarray([1.5], dtype=np.float64),
            64,
            "input_probs must lie in [0, 1]",
        ),
    ),
)
def test_validation_errors_are_preserved(
    weights: npt.NDArray[np.uint64],
    probs: npt.NDArray[np.float64],
    length: int,
    message: str,
) -> None:
    with pytest.raises(ValueError) as captured:
        extension.py_sc_forward_packed(weights, 1, 1, 1, probs, length, 1)
    assert str(captured.value) == message

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — mixed-precision dense engine-binding contracts

"""Installed-extension contracts for the bit-true mixed dense contraction."""

from __future__ import annotations

import importlib

import numpy as np
import numpy.typing as npt
import pytest

from tests.engine_requirement import require_engine

require_engine()
import sc_neurocore_engine as engine

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def _weights() -> npt.NDArray[np.int16]:
    return np.asarray([256, 128], dtype=np.int16)


def _inputs() -> npt.NDArray[np.int32]:
    return np.asarray([512, 1024], dtype=np.int32)


def test_exported_name_signature_and_top_level_identity_are_stable() -> None:
    function = extension.py_mixed_dense_forward_batch_q88_q1616

    assert function.__name__ == "py_mixed_dense_forward_batch_q88_q1616"
    assert function.__module__ == "sc_neurocore_engine.sc_neurocore_engine"
    assert function.__text_signature__ == ("(weights_q88, inputs_q1616, n_outputs, n_inputs)")
    assert engine.py_mixed_dense_forward_batch_q88_q1616 is function


def test_direct_binding_preserves_value_shape_and_dtype() -> None:
    result = extension.py_mixed_dense_forward_batch_q88_q1616(_weights(), _inputs(), 1, 2)

    np.testing.assert_array_equal(result["outputs_q1616"], [1024])
    np.testing.assert_array_equal(result["overflow"], [False])
    np.testing.assert_array_equal(result["underflow"], [False])
    assert result["outputs_q1616"].dtype == np.int32
    assert result["overflow"].dtype == np.bool_
    assert result["underflow"].dtype == np.bool_


@pytest.mark.parametrize(
    ("weights", "inputs", "n_outputs", "n_inputs", "message"),
    (
        (_weights(), _inputs(), 0, 2, "dense shape must have positive inputs and outputs"),
        (
            np.asarray([1, 1], dtype=np.int16),
            np.asarray([1], dtype=np.int32),
            1,
            1,
            "weight length mismatch: expected 1, got 2",
        ),
        (
            np.asarray([1, 1], dtype=np.int16),
            np.asarray([1, 1, 1], dtype=np.int32),
            1,
            2,
            "input length mismatch: expected 2, got 3",
        ),
    ),
)
def test_validation_errors_are_preserved(
    weights: npt.NDArray[np.int16],
    inputs: npt.NDArray[np.int32],
    n_outputs: int,
    n_inputs: int,
    message: str,
) -> None:
    with pytest.raises(ValueError) as captured:
        extension.py_mixed_dense_forward_batch_q88_q1616(
            weights,
            inputs,
            n_outputs,
            n_inputs,
        )
    assert str(captured.value) == message

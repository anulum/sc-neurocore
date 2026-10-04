# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Fixed-point Brunel installed input contracts

"""Exercise constructor refusal and ndarray input contracts at the native boundary."""

from __future__ import annotations

import numpy as np
import pytest

from tests.engine_requirement import require_engine

require_engine()
import sc_neurocore_engine as engine


def _parameters() -> dict[str, object]:
    """Supply a valid four-neuron constructor with read-only CSR arrays."""
    parameters: dict[str, object] = {
        "n_neurons": 4,
        "w_indptr": np.array([0, 1, 2, 3, 4], dtype=np.int64),
        "w_indices": np.array([1, 2, 3, 0], dtype=np.int64),
        "w_data": np.full(4, 128, dtype=np.int16),
        "leak_k": 20,
        "gain_k": 256,
        "ext_lambda": 0.7,
        "ext_weight_fp": 128,
        "seed": 17,
    }
    for value in parameters.values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return parameters


@pytest.mark.parametrize(
    ("field", "values", "message"),
    [
        ("w_indptr", [1, 1, 2, 3, 4], "w_row_offsets must start at 0"),
        ("w_indptr", [0, 2, 1, 3, 4], "w_row_offsets must be nondecreasing"),
        ("w_indptr", [0, 1, 2, 3, 3], "w_row_offsets must end at the number of weights"),
        ("w_indptr", [0, 8, 8, 8, 8], "w_row_offsets must end at the number of weights"),
        ("w_indptr", [0, -1, 2, 3, 4], "w_indptr values must be nonnegative"),
        ("w_indices", [1, 2, 3, 4], "w_col_indices must be less than n_neurons"),
        ("w_indices", [1, 2, -1, 0], "w_indices values must be nonnegative"),
        ("w_data", [128], "w_col_indices len 4 != w_values len 1"),
    ],
)
def test_invalid_csr_refuses_before_running_and_preserves_input_arrays(
    field: str, values: list[int], message: str
) -> None:
    """Invalid signed CSR values refuse without altering caller connectivity."""
    parameters = _parameters()
    array = np.asarray(values, dtype=np.int16 if field == "w_data" else np.int64)
    array.setflags(write=False)
    parameters[field] = array
    before = {
        name: (value.copy(), value.flags.writeable)
        for name, value in parameters.items()
        if isinstance(value, np.ndarray)
    }
    with pytest.raises(ValueError) as refusal:
        engine.FixedPointBrunelNetwork(**parameters)
    assert str(refusal.value) == message
    for name, (original, writeable) in before.items():
        current = parameters[name]
        assert isinstance(current, np.ndarray)
        np.testing.assert_array_equal(current, original)
        assert current.dtype == original.dtype
        assert current.flags.writeable == writeable


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ext_lambda", float("nan"), "ext_lambda must be finite and nonnegative"),
        ("ext_lambda", float("inf"), "ext_lambda must be finite and nonnegative"),
        ("ext_lambda", -float("inf"), "ext_lambda must be finite and nonnegative"),
        ("ext_lambda", -0.1, "ext_lambda must be finite and nonnegative"),
        ("ext_lambda", 1e20, "Invalid ext_lambda:"),
        ("data_width", 0, "data_width must be in [1, 16]"),
        ("data_width", 17, "data_width must be in [1, 16]"),
        ("data_width", 33, "data_width must be in [1, 16]"),
        ("fraction", 16, "fraction must be less than data_width"),
        ("fraction", 32, "fraction must be less than data_width"),
        ("refractory_period", -1, "refractory_period must be nonnegative"),
        ("n_neurons", 2**32, "n_neurons must not exceed u32::MAX"),
    ],
)
def test_invalid_drive_and_fixed_point_configuration_refuse_at_construction(
    field: str, value: int | float, message: str
) -> None:
    """Non-finite drives and unsafe fixed-point parameters fail before stepping."""
    parameters = _parameters()
    parameters[field] = value
    with pytest.raises(ValueError) as refusal:
        engine.FixedPointBrunelNetwork(**parameters)
    assert str(refusal.value).startswith(message)


@pytest.mark.parametrize(
    ("field", "size", "dtype"),
    [("w_indptr", 5, "int64"), ("w_indices", 4, "int64"), ("w_data", 4, "int16")],
)
@pytest.mark.parametrize("layout", ["dtype", "rank", "strided", "reversed", "misaligned"])
def test_numpy_dtype_rank_and_layout_refusals_preserve_input(
    field: str, size: int, dtype: str, layout: str
) -> None:
    """Each public CSR array conversion preserves its measured refusal contract."""
    parameters = _parameters()
    itemsize = np.dtype(dtype).itemsize
    variants = {
        "dtype": np.ones(size, dtype=np.float64),
        "rank": np.ones((1, size), dtype=dtype),
        "strided": np.arange(size * 2, dtype=dtype)[::2],
        "reversed": np.arange(size, dtype=dtype)[::-1],
        "misaligned": np.frombuffer(
            bytearray(size * itemsize + 1), dtype=dtype, count=size, offset=1
        ),
    }
    array = variants[layout]
    array.setflags(write=False)
    parameters[field] = array
    before = (array.tobytes(), array.dtype.str, array.shape, array.strides, array.flags.writeable)
    error = TypeError if layout in {"dtype", "rank"} else ValueError
    with pytest.raises(error) as refusal:
        engine.FixedPointBrunelNetwork(**parameters)
    expected = (
        "'ndarray' object is not an instance of 'ndarray'"
        if error is TypeError
        else f"Cannot read {field}: The given array is not contiguous or is misaligned."
    )
    assert str(refusal.value) == expected
    assert before == (
        array.tobytes(),
        array.dtype.str,
        array.shape,
        array.strides,
        array.flags.writeable,
    )

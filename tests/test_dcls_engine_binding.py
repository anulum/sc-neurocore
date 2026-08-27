# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — DCLS-max engine-binding contracts

"""Installed-extension contracts for the bit-true DCLS-max binding."""

from __future__ import annotations

import importlib
import sys

import numpy as np
import numpy.typing as npt
import pytest

from tests.engine_requirement import require_engine

require_engine()
import sc_neurocore_engine as engine
from sc_neurocore.scpn.dcls_tent_kernel import (
    dcls_max_forward_batch,
    dcls_max_forward_batch_q88,
)

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def _workload() -> tuple[
    npt.NDArray[np.uint8],
    npt.NDArray[np.int16],
    npt.NDArray[np.int16],
    npt.NDArray[np.int16],
    int,
]:
    return (
        np.asarray([1, 1, 1, 0, 1, 0], dtype=np.uint8),
        np.asarray([256, 128, -64, 256, 128, -64], dtype=np.int16),
        np.asarray([256, 256], dtype=np.int16),
        np.asarray([512, 512], dtype=np.int16),
        3,
    )


def test_exported_name_signature_and_top_level_identity_are_stable() -> None:
    function = extension.py_dcls_max_forward_batch_q88

    assert function.__name__ == "py_dcls_max_forward_batch_q88"
    assert function.__module__ == "sc_neurocore_engine.sc_neurocore_engine"
    assert function.__text_signature__ == ("(spikes, weights_q88, centres_q88, sigmas_q88, n_taps)")
    assert engine.py_dcls_max_forward_batch_q88 is function


def test_direct_binding_returns_exact_typed_diagnostics() -> None:
    result = extension.py_dcls_max_forward_batch_q88(*_workload())

    np.testing.assert_array_equal(result["outputs_q88"], [224, 128])
    np.testing.assert_array_equal(result["accumulators_q16_16"], [57_344, 32_768])
    np.testing.assert_array_equal(result["overflow"], [False, False])
    np.testing.assert_array_equal(result["active_tap_counts"], [3, 1])
    np.testing.assert_array_equal(result["max_gates_q88"], [256, 256])
    assert result["outputs_q88"].dtype == np.int16
    assert result["accumulators_q16_16"].dtype == np.int32
    assert result["overflow"].dtype == np.bool_
    assert result["active_tap_counts"].dtype == np.int64
    assert result["max_gates_q88"].dtype == np.int16


def test_production_rust_dispatch_is_bit_exact_with_python_floor() -> None:
    workload = _workload()
    rust = dcls_max_forward_batch(*workload, backend="rust")
    python = dcls_max_forward_batch_q88(*workload)

    for field in (
        "outputs_q88",
        "accumulators_q16_16",
        "overflow",
        "active_tap_counts",
        "max_gates_q88",
    ):
        np.testing.assert_array_equal(getattr(rust, field), getattr(python, field))


def test_saturation_and_shape_failures_are_preserved() -> None:
    saturated = extension.py_dcls_max_forward_batch_q88(
        np.ones(64, dtype=np.uint8),
        np.full(64, 32767, dtype=np.int16),
        np.asarray([0], dtype=np.int16),
        np.asarray([32767], dtype=np.int16),
        64,
    )
    assert saturated["outputs_q88"].tolist() == [32767]
    assert saturated["overflow"].tolist() == [True]

    with pytest.raises(ValueError, match="DCLS forward pass requires at least one tap"):
        extension.py_dcls_max_forward_batch_q88(
            np.empty(0, dtype=np.uint8),
            np.empty(0, dtype=np.int16),
            np.asarray([256], dtype=np.int16),
            np.asarray([512], dtype=np.int16),
            0,
        )


@pytest.mark.parametrize(
    ("n_taps", "error", "message"),
    (
        (-1, OverflowError, "can't convert negative int to unsigned"),
        (1.5, TypeError, "'float' object cannot be interpreted as an integer"),
    ),
)
def test_tap_count_conversion_errors_are_stable(
    n_taps: object, error: type[BaseException], message: str
) -> None:
    spikes, weights, centres, sigmas, _ = _workload()
    with pytest.raises(error) as captured:
        extension.py_dcls_max_forward_batch_q88(spikes, weights, centres, sigmas, n_taps)
    assert str(captured.value) == message
    if sys.version_info >= (3, 11):
        assert captured.value.__notes__ == ["while processing 'n_taps'"]

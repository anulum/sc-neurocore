# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — IQIF engine-binding contracts

"""Installed-extension contracts for the Wu et al. integer QIF binding."""

from __future__ import annotations

import importlib
import sys
from typing import Any

import numpy as np
import pytest

import sc_neurocore_engine as engine
from sc_neurocore.accel import iqif as backends

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")

_PARAMETERS = (0, -64, 64, -32, 1, 1, 127, -128)


def _direct(n_steps: int, current: int) -> tuple[np.ndarray[Any, np.dtype[np.int64]], int, int]:
    trace, spikes, final_v = extension.py_iqif_simulate(*_PARAMETERS, n_steps, current)
    return np.asarray(trace), int(spikes), int(final_v)


def test_exported_name_signature_and_top_level_identity_are_stable() -> None:
    function = extension.py_iqif_simulate

    assert function.__name__ == "py_iqif_simulate"
    assert function.__module__ == "sc_neurocore_engine.sc_neurocore_engine"
    assert function.__text_signature__ == (
        "(v, v_rest, v_threshold, v_reset, a, b, v_max, v_min, n_steps, current)"
    )
    assert engine.py_iqif_simulate is function
    assert "py_iqif_simulate" in engine.__all__


def test_empty_and_integer_trajectory_contracts_are_exact() -> None:
    empty, empty_spikes, empty_final = _direct(0, 0)
    assert empty.shape == (0,)
    assert empty.dtype == np.int64
    assert empty.flags.c_contiguous
    assert (empty_spikes, empty_final) == (0, 0)

    positive, positive_spikes, positive_final = _direct(8, 96)
    np.testing.assert_array_equal(positive, [88, -32, 60, -32, 60, -32, 60, -32])
    assert (positive_spikes, positive_final) == (4, -32)

    negative, negative_spikes, negative_final = _direct(8, -96)
    np.testing.assert_array_equal(negative, [-104, -128, -128, -128, -128, -128, -128, -128])
    assert (negative_spikes, negative_final) == (0, -128)


@pytest.mark.parametrize(
    ("n_steps", "error", "message"),
    (
        (-1, OverflowError, "can't convert negative int to unsigned"),
        (1.5, TypeError, "'float' object cannot be interpreted as an integer"),
    ),
)
def test_step_count_conversion_errors_are_stable(
    n_steps: object, error: type[BaseException], message: str
) -> None:
    with pytest.raises(error) as captured:
        extension.py_iqif_simulate(*_PARAMETERS, n_steps, 0)
    assert str(captured.value) == message
    if sys.version_info >= (3, 11):
        assert captured.value.__notes__ == ["while processing 'n_steps'"]


def test_invalid_parameter_order_keeps_exact_public_error() -> None:
    with pytest.raises(ValueError, match="^invalid IQIF state or parameter ordering$"):
        extension.py_iqif_simulate(0, -64, 64, -32, 1, 1, -129, -128, 1, 0)


def test_production_rust_dispatcher_resolves_the_installed_binding() -> None:
    assert backends._HAS_RUST is True
    assert backends._engine_simulate is engine.py_iqif_simulate

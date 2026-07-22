# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Wilson-Cowan engine-binding contracts

"""Installed-extension contracts for the Wilson-Cowan E/I rate binding."""

from __future__ import annotations

import importlib
from typing import Any, cast

import numpy as np
import pytest

import sc_neurocore_engine as engine

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")

_PARAMETERS = (0.1, 0.05, 10.0, 12.0, 10.0, 3.0, 1.0, 2.0, 1.2, 2.8, 0.05)


def _direct(ext_input: np.ndarray[Any, np.dtype[np.float64]]) -> dict[str, Any]:
    return cast(dict[str, Any], extension.py_wilson_cowan_simulate(*_PARAMETERS, ext_input))


def test_exported_name_signature_and_top_level_identity_are_stable() -> None:
    function = extension.py_wilson_cowan_simulate

    assert function.__name__ == "py_wilson_cowan_simulate"
    assert function.__module__ == "sc_neurocore_engine.sc_neurocore_engine"
    assert function.__text_signature__ == (
        "(e_init, i_init, w_ee, w_ei, w_ie, w_ii, tau_e, tau_i, a, theta, dt, ext_input)"
    )
    assert engine.py_wilson_cowan_simulate is function
    assert "py_wilson_cowan_simulate" in engine.__all__


def test_empty_and_seed_trajectory_contracts_are_exact() -> None:
    empty = _direct(np.empty(0, dtype=np.float64))
    assert tuple(empty) == ("e", "i", "e_final", "i_final")
    for key in ("e", "i"):
        assert empty[key].shape == (0,)
        assert empty[key].dtype == np.float64
        assert empty[key].flags.c_contiguous
    assert empty["e_final"] == 0.1
    assert empty["i_final"] == 0.05

    actual = _direct(np.array([0.0, 0.2, -0.1, 0.3], dtype=np.float64))
    np.testing.assert_array_equal(
        actual["e"],
        np.array(
            [
                0.09601818986717255,
                0.09274551526002495,
                0.08866426690123769,
                0.08585936133859938,
            ]
        ),
    )
    np.testing.assert_array_equal(
        actual["i"],
        np.array(
            [
                0.05005877054685352,
                0.05003311013287667,
                0.049927379960971185,
                0.04975202962653705,
            ]
        ),
    )
    assert actual["e_final"] == actual["e"][-1]
    assert actual["i_final"] == actual["i"][-1]


@pytest.mark.parametrize("index", (0, 6, 10))
def test_invalid_numerical_configuration_keeps_exact_public_error(index: int) -> None:
    arguments: list[object] = [*_PARAMETERS, np.zeros(2, dtype=np.float64)]
    arguments[index] = np.nan if index != 6 else 0.0

    with pytest.raises(ValueError, match="^invalid Wilson-Cowan numerical configuration$"):
        extension.py_wilson_cowan_simulate(*arguments)


def test_input_dtype_and_contiguity_errors_are_stable() -> None:
    with pytest.raises(TypeError, match="'ndarray' object is not an instance of 'ndarray'"):
        extension.py_wilson_cowan_simulate(*_PARAMETERS, np.ones(2, dtype=np.float32))

    non_contiguous = np.arange(4, dtype=np.float64)[::2]
    with pytest.raises(TypeError, match="^The given array is not contiguous or is misaligned\\.$"):
        extension.py_wilson_cowan_simulate(*_PARAMETERS, non_contiguous)

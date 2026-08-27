# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Phi-star engine-binding contracts

"""Installed-extension contracts for the Gaussian Phi-star binding."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from tests.engine_requirement import require_engine

require_engine()
import sc_neurocore_engine as engine
from sc_neurocore.analysis.phi_estimation import phi_star

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def _correlated() -> NDArray[np.float64]:
    generator = np.random.RandomState(7)
    shared = generator.randn(200)
    return np.vstack([shared + 0.3 * generator.randn(200) for _ in range(3)])


def test_exported_name_signature_and_top_level_identity_are_stable() -> None:
    function = extension.py_phi_star

    assert function.__name__ == "py_phi_star"
    assert function.__text_signature__ == "(data, tau)"
    assert engine.py_phi_star is function
    assert "py_phi_star" in engine.__all__
    assert engine._phi_star_rust_available is True


def test_production_rust_backend_uses_the_installed_extension() -> None:
    data = _correlated()
    original = data.copy()

    direct = extension.py_phi_star(data, 1)
    dispatched = phi_star(data, tau=1, backend="rust")

    assert direct == pytest.approx(0.010685405404210493)
    assert dispatched == direct
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize(
    "data",
    (
        np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float64),
        np.empty((0, 3), dtype=np.float64),
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
    ),
)
def test_degenerate_channel_and_time_shapes_preserve_zero_contract(
    data: NDArray[np.float64],
) -> None:
    assert extension.py_phi_star(data, 1) == 0.0


@pytest.mark.parametrize(
    "data",
    (
        np.arange(24, dtype=np.float64).reshape(4, 6)[:, ::2],
        np.asfortranarray(np.arange(24, dtype=np.float64).reshape(4, 6)),
    ),
)
def test_noncontiguous_inputs_preserve_value_error_contract(
    data: NDArray[np.float64],
) -> None:
    with pytest.raises(ValueError, match=r"^py_phi_star requires C-contiguous array input$"):
        extension.py_phi_star(data, 1)


@pytest.mark.parametrize(
    "data",
    (
        np.arange(12, dtype=np.float32).reshape(2, 6),
        np.arange(6, dtype=np.float64),
    ),
)
def test_dtype_and_rank_mismatches_preserve_type_error_contract(
    data: NDArray[np.floating[Any]],
) -> None:
    with pytest.raises(TypeError, match=r"^'ndarray' object is not an instance of 'ndarray'"):
        extension.py_phi_star(data, 1)


@pytest.mark.parametrize(
    ("tau", "error", "message"),
    (
        (-1, OverflowError, r"^can't convert negative int to unsigned"),
        (1.5, TypeError, r"^'float' object cannot be interpreted as an integer"),
    ),
)
def test_tau_conversion_errors_are_stable(
    tau: object, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        extension.py_phi_star(_correlated(), tau)

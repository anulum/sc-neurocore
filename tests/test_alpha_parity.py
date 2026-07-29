# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Python/Rust alpha-synapse parity

"""Compare the modular PyO3 exact-flow batch with the Python golden."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.accel.alpha import simulate_python

engine = pytest.importorskip(
    "sc_neurocore_engine",
    reason="Rust engine wheel not installed",
    exc_type=ImportError,
)
_rust_export = getattr(engine, "py_alpha_simulate", None)
assert _rust_export is not None, "installed Rust engine lacks the Model42 batch export"
py_alpha_simulate = cast(
    "Callable[..., dict[str, Any]]",
    _rust_export,
)

_PARAMETERS = (0.15, 0.08, 0.05, 0.04, 0.03, -0.5, 1.2, 16.0, 4.0, 9.0, 0.5)
_TRACE_KEYS = ("v", "a_exc", "i_exc", "a_inh", "i_inh", "spikes")
_FINAL_KEYS = ("v_final", "a_exc_final", "i_exc_final", "a_inh_final", "i_inh_final")


def _drive(steps: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    index = np.arange(steps, dtype=np.float64)
    return 2.0 + 0.8 * np.sin(index * 0.037), 0.7 + 0.3 * np.cos(index * 0.021)


def _assert_mapping_parity(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    for key in _TRACE_KEYS:
        np.testing.assert_allclose(actual[key], expected[key], rtol=0.0, atol=1.0e-12)
    for key in _FINAL_KEYS:
        assert float(actual[key]) == pytest.approx(float(expected[key]), abs=1.0e-12)
    assert int(actual["spike_count"]) == int(expected["spike_count"])


@pytest.mark.parametrize("steps", (0, 1, 128, 1024))
def test_rust_complete_mapping_matches_python(steps: int) -> None:
    exc, inh = _drive(steps)
    expected = simulate_python(*_PARAMETERS, exc, inh)
    actual = py_alpha_simulate(*_PARAMETERS, exc, inh)
    _assert_mapping_parity(actual, expected)


def test_rust_rejects_invalid_configuration() -> None:
    invalid = (*_PARAMETERS[:-1], -_PARAMETERS[-1])
    exc, inh = _drive(1)
    with pytest.raises(ValueError, match="positive"):
        py_alpha_simulate(*invalid, exc, inh)


def test_rust_rejects_nonfinite_drive() -> None:
    with pytest.raises(ValueError, match="finite"):
        py_alpha_simulate(*_PARAMETERS, np.asarray([1.5, np.nan]), np.asarray([0.1, 0.1]))

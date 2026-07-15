# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Python/Rust Jansen–Rit equation-(6) parity

"""Compare the modular PyO3 batch with the Python golden."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from sc_neurocore.accel.jansen_rit import simulate_python

engine = pytest.importorskip("sc_neurocore_engine", reason="Rust engine wheel not installed")
py_jansen_rit_simulate = engine.py_jansen_rit_simulate

_PARAMETERS = (
    0.1,
    0.2,
    0.3,
    -0.4,
    -0.1,
    0.5,
    3.4,
    21.0,
    95.0,
    55.0,
    128.0,
    2.4,
    5.8,
    0.6,
    0.00012,
)
_TRACE_KEYS = ("y0", "y3", "y1", "y4", "y2", "y5", "eeg")
_STATE_KEYS = _TRACE_KEYS[:6]


def _drive(steps: int) -> np.ndarray:
    index = np.arange(steps, dtype=np.float64)
    return 220.0 + 80.0 * np.sin(index * 0.037) + 20.0 * np.cos(index * 0.011)


def _assert_mapping_parity(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    for key in _TRACE_KEYS:
        np.testing.assert_allclose(actual[key], expected[key], rtol=0.0, atol=1.0e-11)
    for key in _STATE_KEYS:
        assert float(actual[f"{key}_final"]) == pytest.approx(
            float(expected[f"{key}_final"]), abs=1.0e-11
        )


@pytest.mark.parametrize("steps", (0, 1, 128, 1024))
def test_rust_complete_mapping_matches_python(steps: int) -> None:
    expected = simulate_python(*_PARAMETERS, _drive(steps))
    actual = py_jansen_rit_simulate(*_PARAMETERS, _drive(steps))
    _assert_mapping_parity(actual, expected)


def test_rust_rejects_invalid_configuration() -> None:
    invalid = (*_PARAMETERS[:-1], -0.00012)
    with pytest.raises(ValueError, match="positive"):
        py_jansen_rit_simulate(*invalid, _drive(1))


def test_rust_rejects_nonfinite_drive() -> None:
    with pytest.raises(ValueError, match="external drive"):
        py_jansen_rit_simulate(*_PARAMETERS, np.asarray([220.0, np.nan]))

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Python/Rust MPR equation-(12) parity

"""Compare the modular PyO3 batch with the Python golden."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.accel.ermentrout_kopell_pop import simulate_python

engine = pytest.importorskip(
    "sc_neurocore_engine",
    reason="Rust engine wheel not installed",
    exc_type=ImportError,
)
py_ermentrout_kopell_pop_simulate = engine.py_ermentrout_kopell_pop_simulate

_PARAMETERS = (0.13, -1.7, 1.3, 0.8, -4.2, 12.5, 0.004)
_STATE_KEYS = ("r", "v")


def _drive(steps: int) -> npt.NDArray[np.float64]:
    index = np.arange(steps, dtype=np.float64)
    return 1.5 + 0.5 * np.sin(index * 0.037) + 0.25 * np.cos(index * 0.011)


def _assert_mapping_parity(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    for key in _STATE_KEYS:
        np.testing.assert_allclose(actual[key], expected[key], rtol=0.0, atol=1.0e-12)
        assert float(actual[f"{key}_final"]) == pytest.approx(
            float(expected[f"{key}_final"]), abs=1.0e-12
        )


@pytest.mark.parametrize("steps", (0, 1, 128, 1024))
def test_rust_complete_mapping_matches_python(steps: int) -> None:
    expected = simulate_python(*_PARAMETERS, _drive(steps))
    actual = py_ermentrout_kopell_pop_simulate(*_PARAMETERS, _drive(steps))
    _assert_mapping_parity(actual, expected)


def test_rust_rejects_invalid_configuration() -> None:
    invalid = (*_PARAMETERS[:-1], -_PARAMETERS[-1])
    with pytest.raises(ValueError, match="positive"):
        py_ermentrout_kopell_pop_simulate(*invalid, _drive(1))


def test_rust_rejects_nonfinite_drive() -> None:
    with pytest.raises(ValueError, match="external input"):
        py_ermentrout_kopell_pop_simulate(*_PARAMETERS, np.asarray([1.5, np.nan]))

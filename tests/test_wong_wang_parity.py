# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Python/Rust Wong-Wang Euler/OU parity

"""Compare the modular PyO3 batch with the deterministic Python golden."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.accel.wong_wang import simulate_python

engine = pytest.importorskip("sc_neurocore_engine", reason="Rust engine wheel not installed")
py_wong_wang_simulate = engine.py_wong_wang_simulate

_PARAMETERS = (0.24, 0.11, 0.01, -0.02, 0.12, 0.003, 0.7, 0.28, 0.06, 0.31, 0.015, 0.0002)


def _inputs(steps: int) -> tuple[npt.NDArray[np.float64], ...]:
    index = np.arange(steps, dtype=np.float64)
    return (
        0.02 + 0.01 * np.sin(index * 0.07),
        -0.01 + 0.008 * np.cos(index * 0.11),
        np.sin(np.arange(2 * steps, dtype=np.float64) * 0.17),
    )


def _assert_mapping_parity(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    for key in ("s1", "s2", "noise1", "noise2", "r1", "r2"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=0.0, atol=1.0e-12)
    for key in ("s1_final", "s2_final", "noise1_final", "noise2_final"):
        assert float(actual[key]) == pytest.approx(float(expected[key]), abs=1.0e-12)


@pytest.mark.parametrize("steps", (0, 1, 128, 1024))
def test_rust_complete_mapping_matches_python(steps: int) -> None:
    inputs = _inputs(steps)
    expected = simulate_python(*_PARAMETERS, *inputs)
    actual = py_wong_wang_simulate(*_PARAMETERS, *inputs)
    _assert_mapping_parity(actual, expected)


def test_rust_rejects_stimulus_length_mismatch() -> None:
    with pytest.raises(ValueError, match="stim1 and stim2 length mismatch"):
        py_wong_wang_simulate(*_PARAMETERS, np.zeros(2), np.zeros(1), np.zeros(4))


def test_rust_rejects_noise_length_mismatch() -> None:
    with pytest.raises(ValueError, match=r"xi length must be 2 \* n_steps"):
        py_wong_wang_simulate(*_PARAMETERS, np.zeros(2), np.zeros(2), np.zeros(2))


def test_rust_rejects_invalid_configuration() -> None:
    invalid = (*_PARAMETERS[:-1], -0.0002)
    with pytest.raises(ValueError, match="positive"):
        py_wong_wang_simulate(*invalid, *_inputs(1))

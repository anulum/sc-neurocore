# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Python/Rust resonate-and-fire parity

"""Compare the modular PyO3 exact-flow batch with the Python golden."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.accel.resonate_and_fire import simulate_python

engine = pytest.importorskip(
    "sc_neurocore_engine",
    reason="Rust engine wheel not installed",
    exc_type=ImportError,
)
_rust_export = getattr(engine, "py_resonate_and_fire_simulate", None)
if _rust_export is None:
    pytest.skip(
        "installed Rust engine lacks the Model40 batch export",
        allow_module_level=True,
    )
py_resonate_and_fire_simulate = cast(
    "Callable[..., dict[str, Any]]",
    _rust_export,
)

_PARAMETERS = (0.13, -0.27, -0.8, 7.5, 0.9, 0.006)
_TRACE_KEYS = ("x", "y", "spikes")


def _drive(steps: int) -> npt.NDArray[np.float64]:
    index = np.arange(steps, dtype=np.float64)
    return 4.5 + 1.4 * np.sin(index * 0.037) + 0.3 * np.cos(index * 0.011)


def _assert_mapping_parity(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    for key in _TRACE_KEYS:
        np.testing.assert_allclose(actual[key], expected[key], rtol=0.0, atol=1.0e-12)
    assert float(actual["x_final"]) == pytest.approx(float(expected["x_final"]), abs=1.0e-12)
    assert float(actual["y_final"]) == pytest.approx(float(expected["y_final"]), abs=1.0e-12)
    assert int(actual["spike_count"]) == int(expected["spike_count"])


@pytest.mark.parametrize("steps", (0, 1, 128, 1024))
def test_rust_complete_mapping_matches_python(steps: int) -> None:
    expected = simulate_python(*_PARAMETERS, _drive(steps))
    actual = py_resonate_and_fire_simulate(*_PARAMETERS, _drive(steps))
    _assert_mapping_parity(actual, expected)


def test_rust_rejects_invalid_configuration() -> None:
    invalid = (*_PARAMETERS[:-1], -_PARAMETERS[-1])
    with pytest.raises(ValueError, match="positive"):
        py_resonate_and_fire_simulate(*invalid, _drive(1))


def test_rust_rejects_nonfinite_drive() -> None:
    with pytest.raises(ValueError, match="finite"):
        py_resonate_and_fire_simulate(*_PARAMETERS, np.asarray([1.5, np.nan]))

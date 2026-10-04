# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed MAT(1) scalar and array input contracts

"""Exercise actual scalar extraction, array refusal and owning-output recovery."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
import pytest

from tests.engine_requirement import require_engine

extension = require_engine()
import sc_neurocore_engine

FIELDS = (
    "v",
    "theta",
    "refractory_remaining",
    "omega",
    "tau_m",
    "tau_theta",
    "alpha",
    "resistance",
    "refractory_period",
    "dt",
)
DEFAULTS = (0.0, 0.0, 0.0, 19.0, 5.0, 50.0, 37.0, 50.0, 2.0, 0.001)


class NativeCell(Protocol):
    """Describe the installed MAT(1) state interface."""

    def step(self, current: object) -> int:
        """Advance a convertible scalar or refuse without mutation."""
        ...

    def get_state(self) -> dict[str, float]:
        """Return the three detached dynamic state values."""
        ...


_CONSTRUCTOR = cast(Callable[..., NativeCell], sc_neurocore_engine.NonResettingLIFNeuron)
_BATCH = cast(Callable[..., dict[str, object]], extension.py_non_resetting_lif_simulate)


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("value", [None, "0.7", 0.7 + 0j, [0.7]])
def test_each_scalar_constructor_and_batch_field_preserves_extraction_error(
    field: str, value: object
) -> None:
    """Refuse each bad scalar type and preserve a valid subsequent batch."""
    parameters: dict[str, object] = dict(zip(FIELDS, DEFAULTS, strict=True))
    parameters[field] = value
    drive = np.array([0.7, 0.0])
    before = drive.copy()
    expected = f"must be real number, not {type(value).__name__}"
    with pytest.raises(TypeError) as constructor_error:
        _CONSTRUCTOR(**parameters)
    assert str(constructor_error.value) == expected
    with pytest.raises(TypeError) as batch_error:
        _BATCH(**parameters, currents=drive)
    assert str(batch_error.value) == expected
    np.testing.assert_array_equal(drive, before)
    assert _CONSTRUCTOR().step(0.7) == 0
    assert np.asarray(_BATCH(*DEFAULTS, drive)["events"]).shape == (2,)


@pytest.mark.parametrize("value", [None, "0.7", 0.7 + 0j, [0.7]])
def test_step_extraction_refusal_preserves_state_and_next_step(value: object) -> None:
    """Keep the temporal state and recovery transition after failed extraction."""
    cell, control = _CONSTRUCTOR(), _CONSTRUCTOR()
    assert cell.step(0.7) == control.step(0.7)
    before = cell.get_state()
    with pytest.raises(TypeError) as error:
        cell.step(value)
    assert str(error.value) == f"must be real number, not {type(value).__name__}"
    assert cell.get_state() == before
    assert cell.step(0.2) == control.step(0.2)
    assert cell.get_state() == control.get_state()


@pytest.mark.parametrize(
    ("current", "expected"),
    [
        (1, 1.0),
        (True, 1.0),
        (np.float64(0.7), 0.7),
        (np.float32(0.7), float(np.float32(0.7))),
        (np.array(0.7), 0.7),
    ],
)
def test_real_scalar_conversion_matches_explicit_float(current: object, expected: float) -> None:
    """Preserve integer, boolean, NumPy scalar and zero-rank array conversion."""
    cell, control = _CONSTRUCTOR(), _CONSTRUCTOR()
    assert cell.step(current) == control.step(expected)
    assert cell.get_state() == control.get_state()


@pytest.mark.parametrize(
    "layout",
    ["reverse", "broadcast", "misaligned", "swapped", "f32", "i64", "rank0", "rank2", "object"],
)
def test_invalid_array_layout_preserves_input_and_retry(layout: str) -> None:
    """Refuse unsupported alignment, strides, dtype, endianness and rank exactly."""
    drive = np.array([0.7, 0.0, 0.9, 0.2])
    if layout == "reverse":
        invalid = drive[::-1]
    elif layout == "broadcast":
        invalid = np.broadcast_to(drive[:1], (4,))
    elif layout == "misaligned":
        invalid = np.ndarray((4,), dtype=np.float64, buffer=bytearray(33), offset=1)
        invalid[:] = drive
        assert invalid.flags.c_contiguous and not invalid.flags.aligned
    elif layout == "swapped":
        invalid = drive.astype(np.dtype(np.float64).newbyteorder("S"))
    elif layout in ("f32", "i64", "object"):
        invalid = drive.astype({"f32": "f4", "i64": "i8", "object": "O"}[layout])
    elif layout == "rank0":
        invalid = np.array(0.7)
    else:
        invalid = drive[:, None]
    before = invalid.copy()
    with pytest.raises(TypeError) as error:
        _BATCH(*DEFAULTS, invalid)
    expected = (
        "The given array is not contiguous or is misaligned."
        if layout in ("reverse", "broadcast", "misaligned")
        else "'ndarray' object is not an instance of 'ndarray'"
    )
    assert str(error.value) == expected
    np.testing.assert_array_equal(invalid, before)
    assert np.asarray(_BATCH(*DEFAULTS, drive)["events"]).shape == drive.shape
    np.testing.assert_array_equal(drive, [0.7, 0.0, 0.9, 0.2])


@pytest.mark.parametrize("layout", ["empty", "single_reverse", "offset", "readonly"])
def test_contiguous_views_produce_independent_owned_outputs(layout: str) -> None:
    """Accept valid views and return four disjoint writable NumPy owners."""
    owner = np.array([0.7, 0.0, 0.9, 0.2])
    drive = {
        "empty": owner[:0:2],
        "single_reverse": owner[:1][::-1],
        "offset": owner[1:3],
        "readonly": owner,
    }[layout]
    if layout == "readonly":
        drive.setflags(write=False)
    assert drive.flags.c_contiguous and drive.flags.aligned
    result = _BATCH(*DEFAULTS, drive)
    outputs: list[npt.NDArray[np.generic]] = []
    for key in ("voltages", "theta", "refractory", "events"):
        output = np.asarray(result[key])
        assert output.shape == drive.shape
        assert output.dtype == (np.int32 if key == "events" else np.float64)
        assert output.flags.owndata and output.flags.writeable
        assert output.flags.c_contiguous and output.flags.aligned
        assert not np.shares_memory(output, drive)
        outputs.append(output)
    assert not any(np.shares_memory(a, b) for i, a in enumerate(outputs) for b in outputs[i + 1 :])
    np.testing.assert_array_equal(owner, [0.7, 0.0, 0.9, 0.2])
    if layout == "readonly":
        assert not drive.flags.writeable

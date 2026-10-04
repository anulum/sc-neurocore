# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed SC non-resetting adaptive LIF scalar and array input contracts

"""Exercise actual scalar extraction, array layout and post-refusal recovery."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
import pytest

from tests.test_sc_non_resetting_adaptive_lif_engine_binding_configuration import DEFAULTS, FIELDS
from tests.engine_requirement import require_engine

extension = require_engine()
import sc_neurocore_engine


class NativeCell(Protocol):
    """Describe the actual native scalar state interface."""

    def step(self, current: object) -> int:
        """Advance a convertible real scalar or refuse without mutation."""
        ...

    def get_state(self) -> dict[str, float]:
        """Return both native dynamic values as a detached dictionary."""
        ...


_CONSTRUCTOR = cast(Callable[..., NativeCell], sc_neurocore_engine.SCNonResettingAdaptiveLIFNeuron)
_BATCH = cast(Callable[..., dict[str, object]], extension.py_sc_non_resetting_adaptive_lif_simulate)


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("value", [None, "80.0", 80 + 0j, [80.0]])
def test_scalar_constructor_and_batch_extraction_refuse_exact_types(
    field: str, value: object
) -> None:
    """Each exposed field retains its actual extraction error and usable retry."""
    parameters: dict[str, object] = dict(zip(FIELDS, DEFAULTS, strict=True))
    parameters[field] = value
    drive = np.array([80.0, 0.0])
    before = drive.copy()
    expected = f"must be real number, not {type(value).__name__}"
    with pytest.raises(TypeError) as constructor_error:
        _CONSTRUCTOR(**parameters)
    assert str(constructor_error.value) == expected
    with pytest.raises(TypeError) as batch_error:
        _BATCH(**parameters, currents=drive)
    assert str(batch_error.value) == expected
    np.testing.assert_array_equal(drive, before)
    assert _CONSTRUCTOR().step(80.0) == 0
    assert np.asarray(_BATCH(*DEFAULTS, drive)["events"]).shape == (2,)


@pytest.mark.parametrize("value", [None, "80.0", 80 + 0j, [80.0]])
def test_scalar_step_type_error_is_atomic_and_recoverable(value: object) -> None:
    """A failed extraction retains state and the next temporal transition."""
    cell = _CONSTRUCTOR()
    control = _CONSTRUCTOR()
    assert cell.step(80.0) == control.step(80.0)
    before = cell.get_state()
    with pytest.raises(TypeError) as error:
        cell.step(value)
    assert str(error.value) == f"must be real number, not {type(value).__name__}"
    assert cell.get_state() == before
    assert cell.step(20.0) == control.step(20.0)
    assert cell.get_state() == control.get_state()


@pytest.mark.parametrize(
    ("current", "expected"),
    [
        (80, 80.0),
        (True, 1.0),
        (np.float64(80), 80.0),
        (np.float32(80), 80.0),
        (np.array(80.0), 80.0),
    ],
)
def test_supported_real_scalar_conversion_matches_float(current: object, expected: float) -> None:
    """Integer, boolean, NumPy scalars and a zero-dimensional array retain semantics."""
    cell = _CONSTRUCTOR()
    control = _CONSTRUCTOR()
    assert cell.step(current) == control.step(expected)
    assert cell.get_state() == control.get_state()


@pytest.mark.parametrize(
    "layout",
    ["reverse", "broadcast", "misaligned", "swapped", "f32", "i64", "rank0", "rank2", "object"],
)
def test_array_refusal_preserves_input_and_valid_retry(layout: str) -> None:
    """Refuse unsupported memory layouts, endianness, dtypes and ranks exactly."""
    drive = np.array([80.0, 0.0, 120.0, 20.0])
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
        invalid = np.array(80.0)
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
    result = _BATCH(*DEFAULTS, drive)
    assert np.asarray(result["events"]).shape == drive.shape
    np.testing.assert_array_equal(drive, [80.0, 0.0, 120.0, 20.0])


@pytest.mark.parametrize("layout", ["empty", "single_reverse", "offset", "readonly"])
def test_supported_contiguous_views_return_owned_outputs(layout: str) -> None:
    """Accept valid NumPy contiguous views independently of their owner or stride."""
    owner = np.array([80.0, 0.0, 120.0, 20.0])
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
    for key in ("voltages", "theta", "events"):
        output = np.asarray(result[key])
        assert output.shape == drive.shape
        assert output.dtype == (np.int32 if key == "events" else np.float64)
        assert output.flags.owndata and output.flags.writeable
        assert output.flags.c_contiguous and output.flags.aligned
        assert not np.shares_memory(output, drive)
        outputs.append(output)
    assert not any(np.shares_memory(a, b) for i, a in enumerate(outputs) for b in outputs[i + 1 :])
    np.testing.assert_array_equal(owner, [80.0, 0.0, 120.0, 20.0])
    if layout == "readonly":
        assert not drive.flags.writeable

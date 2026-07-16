# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Python/Mojo MPR equation-(12) parity

"""Compare the Mojo shared-library batch with the Python golden."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.accel.ermentrout_kopell_pop import simulate_python
from sc_neurocore.accel.mojo.ermentrout_kopell_pop import (
    _HAS_MOJO_ERMENTROUT_KOPELL_POP,
    simulate_ermentrout_kopell_pop,
)

_PARAMETERS = (0.13, -1.7, 1.3, 0.8, -4.2, 12.5, 0.004)


def _drive(steps: int) -> npt.NDArray[np.float64]:
    index = np.arange(steps, dtype=np.float64)
    return 1.5 + 0.5 * np.sin(index * 0.037)


def test_mojo_shared_library_is_available() -> None:
    assert _HAS_MOJO_ERMENTROUT_KOPELL_POP


@pytest.mark.parametrize("steps", (0, 1, 128, 1024))
def test_mojo_complete_mapping_matches_python(steps: int) -> None:
    expected = simulate_python(*_PARAMETERS, _drive(steps))
    actual = simulate_ermentrout_kopell_pop(*_PARAMETERS, _drive(steps))
    for key in ("r", "v"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=0.0, atol=1.0e-10)


def test_mojo_rejects_nonfinite_drive_before_c_boundary() -> None:
    with pytest.raises(ValueError, match="finite"):
        simulate_ermentrout_kopell_pop(*_PARAMETERS, [1.5, np.inf])


def test_mojo_native_error_is_reported() -> None:
    invalid = (*_PARAMETERS[:-1], -_PARAMETERS[-1])
    with pytest.raises(RuntimeError, match="code 2"):
        simulate_ermentrout_kopell_pop(*invalid, _drive(1))

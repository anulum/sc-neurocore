# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Python/Go Jansen–Rit equation-(6) parity

"""Compare the Go C-shared batch with the Python golden."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.accel.go.jansen_rit import _HAS_GO_JANSEN_RIT, simulate_jansen_rit
from sc_neurocore.accel.jansen_rit import simulate_python

_PARAMETERS = (0.1, 0.2, 0.3, -0.4, -0.1, 0.5, 3.4, 21.0, 95.0, 55.0, 128.0, 2.4, 5.8, 0.6, 0.00012)
_TRACE_KEYS = ("y0", "y3", "y1", "y4", "y2", "y5", "eeg")


def _drive(steps: int) -> np.ndarray:
    index = np.arange(steps, dtype=np.float64)
    return 220.0 + 80.0 * np.sin(index * 0.037)


def test_go_shared_library_is_available() -> None:
    assert _HAS_GO_JANSEN_RIT


@pytest.mark.parametrize("steps", (0, 1, 128, 1024))
def test_go_complete_mapping_matches_python(steps: int) -> None:
    expected = simulate_python(*_PARAMETERS, _drive(steps))
    actual = simulate_jansen_rit(*_PARAMETERS, _drive(steps))
    for key in _TRACE_KEYS:
        np.testing.assert_allclose(actual[key], expected[key], rtol=0.0, atol=1.0e-11)


def test_go_rejects_nonfinite_drive_before_c_boundary() -> None:
    with pytest.raises(ValueError, match="finite"):
        simulate_jansen_rit(*_PARAMETERS, [220.0, np.inf])


def test_go_native_error_is_reported() -> None:
    invalid = (*_PARAMETERS[:-1], -0.00012)
    with pytest.raises(RuntimeError, match="code 2"):
        simulate_jansen_rit(*invalid, _drive(1))

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Python/Go resonate-and-fire parity

"""Compare the Go C-shared exact-flow batch with the Python golden."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.accel.go.resonate_and_fire import (
    _HAS_GO_RESONATE_AND_FIRE,
    simulate_resonate_and_fire,
)
from sc_neurocore.accel.resonate_and_fire import simulate_python

_PARAMETERS = (0.13, -0.27, -0.8, 7.5, 0.9, 0.006)


def _drive(steps: int) -> npt.NDArray[np.float64]:
    index = np.arange(steps, dtype=np.float64)
    return 4.5 + 1.4 * np.sin(index * 0.037)


def test_go_shared_library_is_available() -> None:
    assert _HAS_GO_RESONATE_AND_FIRE


@pytest.mark.parametrize("steps", (0, 1, 128, 1024))
def test_go_complete_mapping_matches_python(steps: int) -> None:
    expected = simulate_python(*_PARAMETERS, _drive(steps))
    actual = simulate_resonate_and_fire(*_PARAMETERS, _drive(steps))
    for key in ("x", "y", "spikes"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=0.0, atol=1.0e-12)
    assert actual["spike_count"] == expected["spike_count"]


def test_go_rejects_nonfinite_drive_before_c_boundary() -> None:
    with pytest.raises(ValueError, match="finite"):
        simulate_resonate_and_fire(*_PARAMETERS, [1.5, np.inf])


def test_go_maps_invalid_native_configuration_to_value_error() -> None:
    invalid = (*_PARAMETERS[:-1], -_PARAMETERS[-1])
    with pytest.raises(ValueError, match="configuration"):
        simulate_resonate_and_fire(*invalid, _drive(1))

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Python/Mojo adaptive-threshold parity

"""Compare the Mojo shared-ABI exact-relaxation batch with the Python golden."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.accel.mojo.adaptive_threshold_if import (
    _HAS_MOJO_ADAPTIVE_THRESHOLD_IF,
    simulate_adaptive_threshold_if,
)
from sc_neurocore.accel.adaptive_threshold_if import simulate_python

_PARAMETERS = (-63.5, -52.5, -68.0, -67.0, -49.0, 4.5, 8.0, 42.0, 0.05)


def _drive(steps: int) -> npt.NDArray[np.float64]:
    index = np.arange(steps, dtype=np.float64)
    return 22.0 + 6.0 * np.sin(index * 0.037)


def test_mojo_shared_library_is_available() -> None:
    assert _HAS_MOJO_ADAPTIVE_THRESHOLD_IF


@pytest.mark.parametrize("steps", (0, 1, 128, 1024))
def test_mojo_complete_mapping_matches_python(steps: int) -> None:
    expected = simulate_python(*_PARAMETERS, _drive(steps))
    actual = simulate_adaptive_threshold_if(*_PARAMETERS, _drive(steps))
    for key in ("v", "theta", "spikes"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=0.0, atol=1.0e-10)
    assert actual["spike_count"] == expected["spike_count"]


def test_mojo_rejects_nonfinite_drive_before_c_boundary() -> None:
    with pytest.raises(ValueError, match="finite"):
        simulate_adaptive_threshold_if(*_PARAMETERS, [1.5, np.inf])


def test_mojo_maps_invalid_native_configuration_to_value_error() -> None:
    invalid = (*_PARAMETERS[:-1], -_PARAMETERS[-1])
    with pytest.raises(ValueError, match="configuration"):
        simulate_adaptive_threshold_if(*invalid, _drive(1))

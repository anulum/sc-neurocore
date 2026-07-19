# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Python/Go alpha-synapse parity

"""Compare the Go C-shared exact-flow batch with the Python golden."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.accel.go.alpha import (
    _HAS_GO_ALPHA,
    simulate_alpha,
)
from sc_neurocore.accel.alpha import simulate_python

_PARAMETERS = (0.15, 0.08, 0.05, 0.04, 0.03, -0.5, 1.2, 16.0, 4.0, 9.0, 0.5)


def _drive(steps: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    index = np.arange(steps, dtype=np.float64)
    return 2.0 + 0.8 * np.sin(index * 0.037), 0.7 + 0.3 * np.cos(index * 0.021)


def test_go_shared_library_is_available() -> None:
    assert _HAS_GO_ALPHA


@pytest.mark.parametrize("steps", (0, 1, 128, 1024))
def test_go_complete_mapping_matches_python(steps: int) -> None:
    exc, inh = _drive(steps)
    expected = simulate_python(*_PARAMETERS, exc, inh)
    actual = simulate_alpha(*_PARAMETERS, exc, inh)
    for key in ("v", "a_exc", "i_exc", "a_inh", "i_inh", "spikes"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=0.0, atol=1.0e-12)
    assert actual["spike_count"] == expected["spike_count"]


def test_go_rejects_nonfinite_drive_before_c_boundary() -> None:
    with pytest.raises(ValueError, match="finite"):
        simulate_alpha(*_PARAMETERS, [1.5, np.inf])


def test_go_maps_invalid_native_configuration_to_value_error() -> None:
    invalid = (*_PARAMETERS[:-1], -_PARAMETERS[-1])
    with pytest.raises(ValueError, match="configuration"):
        simulate_alpha(*invalid, [1.5])

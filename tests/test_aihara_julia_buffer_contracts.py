# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Aihara Julia buffer ownership contracts

"""Exercise the exported Julia batch through its real PythonCall boundary."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pytest

from tests.julia_requirement import require_julia

_BRIDGE = require_julia()

from sc_neurocore.accel import aihara_map

_PARAMETERS = (0.1, 0.7, 1.0, 0.3968, 0.01)


@pytest.fixture(scope="module")
def julia_kernel() -> Any:
    """Use production dispatch to load the maintained, exported Julia module."""
    aihara_map.simulate_aihara_map(current=[], backend="julia")
    return _BRIDGE.Main.AiharaMapNeuronAccel


@pytest.mark.parametrize(("left", "right"), tuple(combinations(range(4), 2)))
@pytest.mark.parametrize("offset", (-1, 0, 1))
def test_aliasing_views_fail_atomically(
    julia_kernel: Any, left: int, right: int, offset: int
) -> None:
    storage = [np.full(5, -123.0) for _ in range(4)]
    storage[0][:] = 0.0
    views = [array[:3] for array in storage]
    if offset < 0:
        views[left] = storage[right][1:4]
    else:
        views[right] = storage[left][offset : offset + 3]
    before = [array.copy() for array in storage]
    with pytest.raises(_BRIDGE.JuliaError) as raised:
        julia_kernel.simulate_aihara_map_b(*_PARAMETERS, *views)
    assert julia_kernel.is_configuration_error(raised.value.exception)
    for actual, expected in zip(storage, before, strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("steps", (0, 1, 64))
def test_adjacent_views_match_complete_python_receipt(julia_kernel: Any, steps: int) -> None:
    storage = np.full(4 * steps, -123.0)
    views = [storage[index * steps : (index + 1) * steps] for index in range(4)]
    views[0][:] = 0.0
    expected = aihara_map.simulate_aihara_map(current=views[0].copy(), backend="python")
    finals = julia_kernel.simulate_aihara_map_b(*_PARAMETERS, *views)
    np.testing.assert_array_equal(views[0], np.zeros(steps))
    for actual, key in zip(views[1:], ("y", "x", "spikes"), strict=True):
        np.testing.assert_allclose(actual, expected[key], rtol=0.0, atol=5e-11)
    for actual, key in zip(finals, ("y_final", "x_final", "spike_count"), strict=True):
        assert actual == pytest.approx(expected[key], rel=0.0, abs=5e-11)

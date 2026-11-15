# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The engine's continuous-attractor ring follows the Python model, ties included

"""Compare the engine's continuous-attractor ring with the Python model.

Both sum each unit's recurrent input left to right, and both put the bump at the first of
equal maxima. Python's built-in ``sum`` compensates rounding from Python 3.12 on, so a model
that summed with it gave other last bits than on Python 3.11 and than the engine. On a ring
driven at 2.0 those bits decided which of several equal units held the bump: after 50 steps
Python 3.12 named unit 0, Python 3.11 unit 7 and the engine unit 12.
"""

from __future__ import annotations

from typing import Any

import pytest

from sc_neurocore.neurons.models.ai_optimized import ContinuousAttractorNeuron

UNITS = 16
DRIVE = 2.0


def _engine_ring() -> Any:
    """Return a 16-unit engine ring, or skip when the engine is not built."""
    engine = pytest.importorskip("sc_neurocore_engine", reason="Rust engine not built")
    return engine.ContinuousAttractorNeuron(UNITS)


def test_a_ring_at_rest_puts_the_bump_at_the_first_unit() -> None:
    """Every unit at 0 is a tie, and both runtimes name unit 0."""
    assert ContinuousAttractorNeuron().bump_position() == 0
    assert _engine_ring().bump_position() == 0


@pytest.mark.parametrize("steps", [1, 50, 200])
def test_a_driven_ring_follows_the_python_model_bit_for_bit(steps: int) -> None:
    """Events, activities and the bump agree on every step at a current of 2.0.

    Parameters
    ----------
    steps : int
        How many steps both rings take.
    """
    rust = _engine_ring()
    python = ContinuousAttractorNeuron()
    for step in range(steps):
        assert rust.step(DRIVE) == python.step(DRIVE), f"event differs at step {step}"
        activities = [float(value) for value in rust.get_state()["u"]]
        assert activities == python.u, f"activities differ at step {step}"
        assert rust.bump_position() == python.bump_position(), f"bump differs at step {step}"


def test_the_recurrent_input_is_summed_left_to_right() -> None:
    """A compensated sum would give another step, so the model does not depend on the Python version.

    With every activity at 1 each rate is 0.5, and the three weights give the terms 5e15, 0.5
    and -5e15. Left to right, 5e15 + 0.5 rounds back to 5e15 and the input is 0; a compensated
    sum gives 0.5. The step is then 1 + (-1 + 0) / 10 = 0.9 for every unit, where a
    compensated sum would give 0.95.
    """
    weights = [[1e16, 1.0, -1e16] for _ in range(3)]
    ring = ContinuousAttractorNeuron(n_units=3, u=[1.0, 1.0, 1.0], _weights=weights)
    ring.step(0.0)
    assert ring.u == [1.0 + (-1.0 + 0.0 + 0.0) / 10.0 * 1.0] * 3


@pytest.mark.parametrize(
    ("activities", "position"),
    [
        ([0.0, 3.0, 1.0, 3.0], 1),
        ([1.0, 2.0, 0.5, 2.0, 1.5], 1),
        ([0.2] * UNITS, 0),
        ([1.0, float("nan"), 2.0, 0.0], 2),
        ([float("nan"), 1.0, 2.0, 0.0], 0),
    ],
)
def test_the_python_model_takes_the_first_of_equal_maxima(
    activities: list[float], position: int
) -> None:
    """The rule the engine's own tests hold: the first of equal maxima, and a NaN never takes the bump.

    A NaN holds the position only as unit 0, since nothing is more active than it.

    Parameters
    ----------
    activities : list[float]
        The ring's activities.
    position : int
        The unit the bump must be at.
    """
    ring = ContinuousAttractorNeuron(n_units=len(activities), u=list(activities))
    assert ring.bump_position() == position

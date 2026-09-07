# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — What a projection's delay means, and what it runs

"""The delay a projection runs is not always the delay it was asked for.

``Projection`` takes a delay in **timesteps** and rounds a non-integral request
to whole steps. That was undocumented and unpinned, and the object reported the
request rather than the rounding, so a run could be recorded with a delay it
never used — a Brunel configuration asking for 1.5 steps runs 2.

These cases pin the rounding, including its half-to-even edge, and hold
``delay_steps`` to reporting what the buffers were actually built for.
"""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.network import Population, Projection


def _projection(delay: float | np.ndarray[object, object]) -> Projection:
    """Build a small all-to-all projection with the given delay.

    Parameters
    ----------
    delay : float or numpy.ndarray
        The delay to request, in timesteps.

    Returns
    -------
    Projection
        The projection.
    """
    source = Population("SCLapicqueLIFNeuron", 3, label="src")
    target = Population("SCLapicqueLIFNeuron", 3, label="tgt")
    return Projection(source, target, weight=1.0, probability=1.0, delay=delay, seed=1)


class TestUniformDelayRounding:
    @pytest.mark.parametrize(
        ("requested", "executed"),
        [
            (0, 0),
            (1.0, 1),
            (2.0, 2),
            (3.7, 4),
            (0.4, 1),
            (1.5, 2),
            (2.5, 2),
        ],
    )
    def test_the_executed_delay_is_whole_steps(self, requested: float, executed: int) -> None:
        assert _projection(requested).delay_steps == executed

    def test_a_half_step_rounds_to_even_so_one_and_a_half_and_two_and_a_half_agree(self) -> None:
        # Not a curiosity: 1.5 and 2.5 are a step apart and run identically.
        assert _projection(1.5).delay_steps == _projection(2.5).delay_steps

    def test_a_delay_smaller_than_half_a_step_still_occupies_one(self) -> None:
        # A spike that is delayed at all cannot arrive in the step it was
        # emitted, so 0.4 is one step rather than none.
        assert _projection(0.4).delay_steps == 1

    def test_no_delay_occupies_no_steps(self) -> None:
        assert _projection(0).delay_steps == 0

    def test_the_requested_value_is_kept_and_differs_from_what_runs(self) -> None:
        # Both are readable on purpose. Recording the wrong one is how a run
        # gets published with a delay it did not use.
        projection = _projection(1.5)

        assert projection.delay == pytest.approx(1.5)
        assert projection.delay_steps == 2

    def test_the_buffer_is_built_for_the_executed_delay(self) -> None:
        projection = _projection(3.7)

        assert projection.max_delay == projection.delay_steps


class TestPerSynapseDelays:
    def test_each_synapse_is_rounded_to_whole_steps(self) -> None:
        projection = _projection(np.array([0.4, 1.5, 2.5, 3.7] + [1.0] * 5, dtype=np.float64))

        assert list(np.asarray(projection.delay_steps)[:4]) == [0, 2, 2, 4]

    def test_the_maximum_is_the_largest_executed_delay(self) -> None:
        projection = _projection(np.array([0.4, 1.5, 2.5, 3.7] + [1.0] * 5, dtype=np.float64))

        assert projection.max_delay == 4

    def test_a_length_mismatch_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must match"):
            _projection(np.array([1.0, 2.0], dtype=np.float64))

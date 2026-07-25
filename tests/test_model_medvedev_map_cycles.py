# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Medvedev cycle and batch contracts

"""Golden-cycle, batch-parity, and stability tests for the Medvedev map."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from sc_neurocore.neurons.models.medvedev_map import MedvedevMapNeuron


def test_zero_current_golden_cycle() -> None:
    """The source map reproduces its calibrated 100-step orbit."""
    trace, events = MedvedevMapNeuron().simulate(100, 0.0, backend="python")
    assert events == 100
    assert trace[-1] == pytest.approx(0.19448491761002404, abs=1e-15)
    assert float(np.mean(trace)) == pytest.approx(0.21623098362239998, abs=1e-15)
    assert np.unique(trace).size == 7


def test_driven_golden_cycle_and_event_vector() -> None:
    """The maintained I=2 protocol has a four-state, 75-event cycle."""
    trace, events = MedvedevMapNeuron().simulate(100, 2.0, backend="python")
    expected_cycle = np.array(
        [
            0.20201527871456648,
            0.23396543697847846,
            0.26318342915295445,
            0.2514078836724436,
        ]
    )
    assert events == 75
    np.testing.assert_array_equal(trace[:4], expected_cycle)
    np.testing.assert_array_equal(trace, np.tile(expected_cycle, 25))


def test_reproducibility_hash_is_stable() -> None:
    """The descriptor's 1000-step little-endian trace hash is exact."""
    trace, events = MedvedevMapNeuron().simulate(1000, 2.0, backend="python")
    digest = hashlib.sha256(trace.astype("<f8", copy=False).tobytes()).hexdigest()
    assert events == 750
    assert digest == "4e45193f652b8c4ab1fc860b179585a52c565cfbe1769b17e850ab770a232f2c"


def test_batch_matches_repeated_checked_steps() -> None:
    """The batch surface and single-step surface commit the same recurrence."""
    batch = MedvedevMapNeuron()
    trace, events = batch.simulate(300, 2.0, backend="python")
    stepper = MedvedevMapNeuron()
    manual = []
    manual_events = 0
    for _step in range(300):
        manual_events += stepper.step(2.0)
        manual.append(stepper.u)
    np.testing.assert_array_equal(trace, np.asarray(manual, dtype=np.float64))
    assert events == manual_events
    assert batch.u == stepper.u


@pytest.mark.parametrize("current", (0.0, 2.0, 16.0, 1024.0))
def test_long_run_remains_finite(current: float) -> None:
    """The enrolled operating envelope never commits non-finite state."""
    trace, _events = MedvedevMapNeuron().simulate(10_000, current, backend="python")
    assert np.isfinite(trace).all()

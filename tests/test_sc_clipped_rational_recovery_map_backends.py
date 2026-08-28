# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Retained rational-recovery runtime parity

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.neurons.models import sc_clipped_rational_recovery_map
from sc_neurocore.neurons.models.sc_clipped_rational_recovery_map import (
    SCClippedRationalRecoveryMapNeuron,
)


def _run(
    backend: str, *, steps: int = 512, current: float = 0.0
) -> tuple[npt.NDArray[np.float64], int, float, float]:
    neuron = SCClippedRationalRecoveryMapNeuron()
    trace, events = neuron.simulate(steps, current, backend)
    return trace, events, neuron.x, neuron.y


@pytest.mark.parametrize("backend", ["rust", "julia", "go", "mojo"])
def test_binary64_runtime_parity(backend: str) -> None:
    available = {
        "rust": sc_clipped_rational_recovery_map._HAS_RUST,
        "julia": sc_clipped_rational_recovery_map._ensure_julia_loaded(),
        "go": sc_clipped_rational_recovery_map._ensure_go_loaded(),
        "mojo": sc_clipped_rational_recovery_map._ensure_mojo_loaded(),
    }[backend]
    assert available
    expected = _run("python")
    observed = _run(backend)
    np.testing.assert_array_equal(observed[0], expected[0])
    assert observed[1:] == expected[1:]


def test_mojo_one_step_and_event_contract() -> None:
    assert sc_clipped_rational_recovery_map._ensure_mojo_loaded()
    rng = np.random.default_rng(20260828)
    worst = 0.0
    for _ in range(2_000):
        x = float(rng.uniform(-2.0, 2.0))
        y = float(rng.uniform(-2.0, 2.0))
        current = float(rng.uniform(-0.5, 0.5))
        reference = SCClippedRationalRecoveryMapNeuron(x=x, y=y)
        observed = SCClippedRationalRecoveryMapNeuron(x=x, y=y)
        ref_trace, ref_event = reference.simulate(1, current, "python")
        got_trace, got_event = observed.simulate(1, current, "mojo")
        assert got_event == ref_event
        worst = max(
            worst,
            abs(float(got_trace[0]) - float(ref_trace[0])),
            abs(observed.y - reference.y),
        )
    assert worst == 0.0


def test_validation_is_failure_atomic() -> None:
    with pytest.raises(ValueError, match="integer"):
        SCClippedRationalRecoveryMapNeuron().simulate(True)
    with pytest.raises(ValueError, match="between"):
        SCClippedRationalRecoveryMapNeuron().simulate(-1)
    with pytest.raises(ValueError, match="finite"):
        SCClippedRationalRecoveryMapNeuron().step(float("nan"))
    neuron = SCClippedRationalRecoveryMapNeuron()
    before = (neuron.x, neuron.y)
    with pytest.raises(ValueError, match="positive"):
        neuron.alpha = 0.0
        neuron.step(0.0)
    assert (neuron.x, neuron.y) == before


def test_simulate_matches_repeated_step() -> None:
    trace, events = SCClippedRationalRecoveryMapNeuron().simulate(128, backend="python")
    stepper = SCClippedRationalRecoveryMapNeuron()
    manual = []
    manual_events = 0
    for _ in range(128):
        manual_events += stepper.step(0.0)
        manual.append(stepper.x)
    np.testing.assert_array_equal(trace, np.asarray(manual))
    assert events == manual_events

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Model 50 dual-identity PyO3 stateful binding contracts

from __future__ import annotations

import numpy as np

from tests.engine_requirement import require_engine

require_engine()
import sc_neurocore_engine as engine
from sc_neurocore.neurons.models.non_resetting_lif import NonResettingLIFNeuron
from sc_neurocore.neurons.models.sc_non_resetting_adaptive_lif import (
    SCNonResettingAdaptiveLIFNeuron,
)


def test_source_mat1_engine_preserves_non_resetting_trace() -> None:
    currents = [0.0] * 32 + [0.7] * 8192 + [0.2, 0.9] * 1024
    hand = NonResettingLIFNeuron()
    native = engine.NonResettingLIFNeuron()
    hand_events = [hand.step(current) for current in currents]
    native_events = [native.step(current) for current in currents]
    np.testing.assert_array_equal(native_events, hand_events)
    state = native.get_state()
    assert set(state) == {"v", "theta", "refractory_remaining"}
    np.testing.assert_allclose(
        tuple(state[key] for key in ("v", "theta", "refractory_remaining")),
        (hand.v, hand.theta, hand.refractory_remaining),
        rtol=0.0,
        atol=2.0e-12,
    )
    assert sum(native_events) == 1


def test_sc_engine_preserves_frozen_project_trace() -> None:
    currents = [0.0] * 32 + [20.0] * 96 + [20.0, 60.0] * 64
    hand = SCNonResettingAdaptiveLIFNeuron()
    native = engine.SCNonResettingAdaptiveLIFNeuron()
    hand_events = [hand.step(current) for current in currents]
    native_events = [native.step(current) for current in currents]
    np.testing.assert_array_equal(native_events, hand_events)
    state = native.get_state()
    assert set(state) == {"v", "theta"}
    np.testing.assert_allclose(
        tuple(state[key] for key in ("v", "theta")),
        (hand.v, hand.theta),
        rtol=0.0,
        atol=2.0e-12,
    )
    assert sum(native_events) == 5

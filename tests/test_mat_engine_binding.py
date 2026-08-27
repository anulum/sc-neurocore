# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — MAT dual-identity PyO3 binding contracts

"""Exercise the stateful Rust engine surfaces for both MAT identities."""

from __future__ import annotations

import numpy as np

from tests.engine_requirement import require_engine

require_engine()
import sc_neurocore_engine as engine
from sc_neurocore.neurons.models.mat import MATNeuron
from sc_neurocore.neurons.models.sc_resetting_mat import SCResettingMATNeuron


def test_source_mat_engine_preserves_non_resetting_trace() -> None:
    currents = np.full(5000, 0.7)
    hand = MATNeuron()
    native = engine.MATNeuron()
    hand_events = [hand.step(float(current)) for current in currents]
    native_events = [native.step(float(current)) for current in currents]
    np.testing.assert_array_equal(native_events, hand_events)
    state = native.get_state()
    assert set(state) == {"v", "theta1", "theta2", "refractory_remaining"}
    np.testing.assert_allclose(
        tuple(state[key] for key in ("v", "theta1", "theta2", "refractory_remaining")),
        (hand.v, hand.theta1, hand.theta2, hand.refractory_remaining),
        rtol=0.0,
        atol=2.0e-12,
    )
    assert hand.v != 0.0


def test_sc_resetting_mat_engine_preserves_project_anchor() -> None:
    currents = [0.0] * 32 + [50.0] * 96 + [20.0, 60.0] * 64
    hand = SCResettingMATNeuron()
    native = engine.SCResettingMATNeuron()
    hand_events = [hand.step(current) for current in currents]
    native_events = [native.step(current) for current in currents]
    np.testing.assert_array_equal(native_events, hand_events)
    state = native.get_state()
    assert set(state) == {"v", "theta1", "theta2"}
    np.testing.assert_allclose(
        tuple(state[key] for key in ("v", "theta1", "theta2")),
        (hand.v, hand.theta1, hand.theta2),
        rtol=0.0,
        atol=2.0e-12,
    )
    assert sum(native_events) == 13

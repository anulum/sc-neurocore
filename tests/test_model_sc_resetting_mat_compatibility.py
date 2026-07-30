# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - historical resetting-MAT compatibility anchor

from __future__ import annotations

import hashlib
import struct

import pytest

from sc_neurocore.neurons.models.sc_resetting_mat import SCResettingMATNeuron


def test_historical_trace_is_bit_identical_after_identity_split() -> None:
    """The renamed SC model preserves the audited 256-step binary64 trace."""
    currents = [0.0] * 32 + [50.0] * 96 + [20.0, 60.0] * 64
    neuron = SCResettingMATNeuron()
    trace = bytearray()
    events = 0
    for current in currents:
        spike = neuron.step(current)
        events += spike
        trace.extend(struct.pack("<dddB", neuron.v, neuron.theta1, neuron.theta2, spike))

    assert events == 13
    assert neuron.v == -70.0
    assert neuron.theta1 == pytest.approx(5.262135955944077, abs=1.0e-15)
    assert neuron.theta2 == pytest.approx(21.149478444493045, abs=1.0e-15)
    assert hashlib.sha256(trace).hexdigest() == (
        "b64411c28f4ab24e87fb52a115fd9379793412350af57a933806f1b6c32af259"
    )

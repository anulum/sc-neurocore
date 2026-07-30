# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Frozen compatibility receipt for the retained SC project recurrence."""

from __future__ import annotations

import hashlib
import struct

from sc_neurocore.neurons.models.sc_non_resetting_adaptive_lif import (
    SCNonResettingAdaptiveLIFNeuron,
)


def test_historical_project_trace_is_bit_identical() -> None:
    """Preserve the exact pre-split 256-step state/event trace."""
    inputs = [0.0] * 32 + [20.0] * 96 + [value for _ in range(64) for value in (20.0, 60.0)]
    neuron = SCNonResettingAdaptiveLIFNeuron()
    digest = hashlib.sha256()
    events = 0
    for current in inputs:
        event = neuron.step(current)
        events += event
        digest.update(struct.pack("<ddi", neuron.v, neuron.theta, event))
    assert events == 5
    assert (neuron.v, neuron.theta) == (-32.61772042832371, -27.97424372241646)
    assert digest.hexdigest() == "7dd9f76fd1d819bc462460112cfb5906b137935db466bfd60e206f1b4303ae25"

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared typing helpers for neuron integrator path tests

from __future__ import annotations

from typing import Any, Protocol, cast


class SpikeSteppable(Protocol):
    """Neuron interface required by the integrator spike-count assertions."""

    def step(self, current: float) -> int:
        """Advance one step and report whether a spike occurred."""


def count_spikes(neuron: SpikeSteppable, current: float, steps: int) -> int:
    """Count spikes emitted by one neuron over a fixed stimulus window."""

    return sum(int(neuron.step(current)) for _ in range(steps))


def as_untyped(value: object) -> Any:
    """Cross a deliberate negative-test typing boundary."""

    return cast(Any, value)

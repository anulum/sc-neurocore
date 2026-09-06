# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Complete-packet event domain across every model boundary

"""Every model that accepts a foreign complete packet refuses the same events.

Each model with a Rust, Julia, Go or Mojo complete-packet lane validates what
comes back before it mutates its own state. Six of the seven checked the event
array as it arrived; ``QuadraticIFNeuron`` cast it to ``uint8`` first and then
looked at the result. Measured on its real validator before the change: an event
array containing ``256`` was accepted and became ``[0, 0, 0, 0]`` — a foreign
spike turned into a silent non-spike — and ``0.5`` was accepted the same way,
while ``-1`` and ``2`` were refused. Its six siblings refused all four.

These cases pin the domain for every boundary at once, so one validator cannot
drift away from the others again. Each packet is produced by the model's own
Python path, so everything except the substituted event array is genuinely
valid and a refusal cannot be a false positive from some other field.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pytest

from sc_neurocore.neurons.models.adex import AdExNeuron
from sc_neurocore.neurons.models.dpi_neuron import DPINeuron
from sc_neurocore.neurons.models.expif import ExpIFNeuron
from sc_neurocore.neurons.models.lapicque import LapicqueNeuron
from sc_neurocore.neurons.models.perfect_integrator import PerfectIntegratorNeuron
from sc_neurocore.neurons.models.quadratic_if import QuadraticIFNeuron
from sc_neurocore.neurons.models.theta import ThetaNeuron

_STEPS = 6


class _CompletePacketModel(Protocol):
    """The surface every model with a foreign complete-packet lane exposes.

    The packet shapes differ per model — a two-state model returns two traces
    where a four-state model returns four — so the arguments stay untyped here
    and the boundary table supplies them.
    """

    def _simulate_python_complete(self, n_steps: int, *arguments: Any) -> Any:
        """Return a valid packet from the model's own Python recurrence."""

    @staticmethod
    def _validated_complete_packet(packet: Any, n_steps: int, *arguments: Any) -> Any:
        """Validate a foreign packet before the model mutates its own state."""


@dataclass(frozen=True, slots=True)
class _Boundary:
    """One model's complete-packet boundary, as this contract drives it."""

    name: str
    model: type[_CompletePacketModel]
    run_arguments: tuple[float, ...]
    events_at: int
    validator_arguments: Callable[[Any], tuple[Any, ...]] = lambda neuron: ()


_BOUNDARIES: tuple[_Boundary, ...] = (
    _Boundary("AdExNeuron", AdExNeuron, (400.0,), 2),
    _Boundary("DPINeuron", DPINeuron, (5e-10,), 3),
    _Boundary("ExpIFNeuron", ExpIFNeuron, (400.0,), 2),
    _Boundary("LapicqueNeuron", LapicqueNeuron, (5.0,), 1),
    _Boundary("PerfectIntegratorNeuron", PerfectIntegratorNeuron, (5.0,), 1),
    _Boundary("QuadraticIFNeuron", QuadraticIFNeuron, (5.0,), 1),
    _Boundary("ThetaNeuron", ThetaNeuron, (5.0,), 1),
)

#: Event arrays no boundary may accept. ``256`` and ``0.5`` are the ones a cast
#: to ``uint8`` silently turns into a non-spike; ``-1`` and ``2`` are outside the
#: domain by inspection.
_REJECTED: tuple[tuple[str, Any], ...] = (
    ("wraps to zero under uint8", np.full(_STEPS, 256, dtype=np.int64)),
    ("truncates to zero", np.full(_STEPS, 0.5, dtype=np.float64)),
    ("negative", np.full(_STEPS, -1, dtype=np.int64)),
    ("above the domain", np.full(_STEPS, 2, dtype=np.int64)),
)


def _validate(boundary: _Boundary, packet: tuple[Any, ...]) -> Any:
    """Drive one boundary's validator with the arguments it requires."""
    neuron = boundary.model()
    extra = boundary.validator_arguments(neuron)
    return boundary.model._validated_complete_packet(packet, _STEPS, *extra)


def _packet(boundary: _Boundary) -> tuple[Any, ...]:
    """Return a genuinely valid packet from the model's own Python path."""
    neuron = boundary.model()
    return tuple(neuron._simulate_python_complete(_STEPS, *boundary.run_arguments))


def _events_of(result: Any) -> np.ndarray[Any, Any]:
    """Return the single validated event array a result carries.

    Models place it at different depths, so it is found by kind rather than by
    position: exactly one ``uint8`` array is what a validated packet returns.
    """
    found: list[np.ndarray[Any, Any]] = []

    def walk(value: Any) -> None:
        if isinstance(value, np.ndarray):
            if value.dtype == np.uint8:
                found.append(value)
            return
        if isinstance(value, tuple):
            for item in value:
                walk(item)

    walk(result)
    assert len(found) == 1, f"expected one uint8 event array, found {len(found)}"
    return found[0]


def _with_events(packet: tuple[Any, ...], index: int, events: Any) -> tuple[Any, ...]:
    """Return the packet with only its event array replaced."""
    replaced = list(packet)
    replaced[index] = events
    return tuple(replaced)


@pytest.mark.parametrize("boundary", _BOUNDARIES, ids=[row.name for row in _BOUNDARIES])
class TestTheEventDomain:
    def test_a_valid_packet_is_accepted(self, boundary: _Boundary) -> None:
        """Without this the refusals below could all be false positives."""
        events = _events_of(_validate(boundary, _packet(boundary)))

        assert set(events.tolist()) <= {0, 1}

    @pytest.mark.parametrize(("reason", "events"), _REJECTED, ids=[row[0] for row in _REJECTED])
    def test_an_event_outside_the_domain_is_refused(
        self, boundary: _Boundary, reason: str, events: Any
    ) -> None:
        packet = _with_events(_packet(boundary), boundary.events_at, events)

        with pytest.raises((RuntimeError, FloatingPointError, ValueError)) as raised:
            _validate(boundary, packet)

        assert "event" in str(raised.value).lower(), (
            f"{boundary.name} refused for another reason: {raised.value}"
        )


class TestTheBoundaryCensus:
    def test_every_model_with_a_complete_packet_lane_is_covered(self) -> None:
        """A new lane must join this contract rather than define its own."""
        from pathlib import Path

        models = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/models"
        with_boundary = {
            path.stem
            for path in models.glob("*.py")
            if "_validated_complete_packet" in path.read_text(encoding="utf-8")
        }
        covered = {boundary.model.__module__.rsplit(".", 1)[-1] for boundary in _BOUNDARIES}

        assert with_boundary == covered, (
            "a model gained or lost a complete-packet boundary; add it to _BOUNDARIES "
            "with the arguments its Python path and its validator need, so the new "
            f"boundary joins this contract instead of defining its own: {with_boundary ^ covered}"
        )

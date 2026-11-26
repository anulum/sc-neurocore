# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Parametric LIF: the engine returns the spike of the updated potential

"""The engine's Parametric LIF returns the same spike and potential as the Python model.

Fang et al. (2021, Eq. 2-3) charge ``H[t] = sigmoid(a) * V[t-1] + X[t]`` and fire
``S[t] = Theta(H[t] - threshold)``; the Python model, the Go, Julia and safety-Rust lanes and
the model page all return that ``S[t]``. The engine returned the spike of the stored potential
before the update, so its spike train trailed the Python one by a step: an input of 5.0 from
rest spiked at once in Python and only on the following step in the engine.
"""

from __future__ import annotations

from typing import Protocol, cast

import numpy as np

from sc_neurocore.neurons.models.plif import ParametricLIFNeuron
from tests.engine_requirement import require_engine

engine = require_engine()


class _EngineNeuron(Protocol):
    """The engine binding's stepping interface."""

    def step(self, current: float) -> int: ...

    def reset(self) -> None: ...

    def get_state(self) -> dict[str, float]: ...


def _pair() -> tuple[_EngineNeuron, ParametricLIFNeuron]:
    """Return a fresh engine neuron and a fresh Python neuron, both at their defaults."""
    return cast(_EngineNeuron, engine.ParametricLIFNeuron()), ParametricLIFNeuron()


def test_a_suprathreshold_input_from_rest_spikes_on_the_same_step() -> None:
    """From rest, an input of 5.0 spikes on the first step in both, with the same potential."""
    rust, python = _pair()
    assert rust.step(5.0) == python.step(5.0) == 1
    assert rust.get_state()["v"] == python.v == 5.0


def test_the_documented_sequence_spikes_and_resets_identically() -> None:
    """5.0, 0.5 and 0.75 give spikes 1, 0, 1 and potentials 5.0, 0.5, 1.0 in both."""
    rust, python = _pair()
    for current, spike, potential in ((5.0, 1, 5.0), (0.5, 0, 0.5), (0.75, 1, 1.0)):
        assert rust.step(current) == python.step(current) == spike
        assert rust.get_state()["v"] == python.v == potential


def test_every_step_of_a_mixed_drive_matches_the_python_model() -> None:
    """Over 2000 seeded sub- and suprathreshold inputs, every spike and potential agree."""
    rust, python = _pair()
    drive = np.random.default_rng(20211011).uniform(-0.5, 2.0, size=2000)
    rust_spikes, python_spikes = [], []
    for step, current in enumerate(drive.tolist()):
        rust_spikes.append(rust.step(current))
        python_spikes.append(python.step(current))
        assert rust_spikes[-1] == python_spikes[-1], step
        assert rust.get_state()["v"] == python.v, step
    assert sum(python_spikes) > 0
    assert sum(rust_spikes) == sum(python_spikes)


def test_reset_returns_both_to_rest() -> None:
    """After a spiking run, reset puts both potentials back to zero and they agree again."""
    rust, python = _pair()
    for _ in range(5):
        rust.step(1.5)
        python.step(1.5)
    rust.reset()
    python.reset()
    assert rust.get_state()["v"] == python.v == 0.0
    assert rust.step(1.5) == python.step(1.5)

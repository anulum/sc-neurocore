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
rest spiked at once in Python and only on the following step in the engine. The engine also
took ``sigmoid(a)`` as ``1 / (1 + exp(-a))`` for every ``a``, where the other lanes use
``exp(a) / (1 + exp(a))`` below zero, and its class took no ``a`` at all.
"""

from __future__ import annotations

import copy
import math
import pickle
from typing import Any, Protocol, cast

import numpy as np
import pytest

from sc_neurocore.neurons.models.plif import ParametricLIFNeuron
from tests.engine_requirement import require_engine

engine = require_engine()


class _EngineNeuron(Protocol):
    """The engine binding's stepping interface."""

    def step(self, current: float) -> int: ...

    def reset(self) -> object: ...

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


@pytest.mark.parametrize("a", [-3.0, -0.25, 1.5])
def test_a_non_default_decay_parameter_matches_the_python_model(a: float) -> None:
    """With ``a`` set on both, every step of the 2000-step drive agrees bit for bit.

    Below zero the engine took ``1 / (1 + exp(-a))`` where the Python model takes
    ``exp(a) / (1 + exp(a))``; the two differ in the last bit for about half of ``a`` in
    -20..0, ``a = -3`` among them, and the potentials then drifted apart.

    Parameters
    ----------
    a : float
        Decay parameter given to both neurons.
    """
    rust = cast(_EngineNeuron, engine.ParametricLIFNeuron(a=a))
    python = ParametricLIFNeuron(a=a)
    drive = np.random.default_rng(20211011).uniform(-0.5, 2.0, size=2000)
    spikes = 0
    for step, current in enumerate(drive.tolist()):
        fired = python.step(current)
        assert rust.step(current) == fired, step
        assert rust.get_state()["v"] == python.v, step
        spikes += fired
    assert spikes > 0


def test_the_engine_class_takes_and_checks_the_python_fields() -> None:
    """``v``, ``a`` and ``threshold`` reach the engine and are refused as the Python model refuses them."""
    rust = cast(_EngineNeuron, engine.ParametricLIFNeuron(v=0.5, a=-1.0, threshold=2.0))
    python = ParametricLIFNeuron(v=0.5, a=-1.0, threshold=2.0)
    assert rust.get_state()["v"] == python.v == 0.5
    assert rust.step(1.9) == python.step(1.9)
    assert rust.get_state()["v"] == python.v
    for arguments in (
        {"v": math.nan},
        {"a": math.inf},
        {"threshold": 0.0},
        {"threshold": math.nan},
    ):
        with pytest.raises(ValueError):
            engine.ParametricLIFNeuron(**arguments)
        with pytest.raises(ValueError):
            ParametricLIFNeuron(**arguments)


_NAMESPACE = "sc_neurocore_engine.sc_neurocore_engine"


def _make() -> _EngineNeuron:
    """Return an engine PLIF with a non-default decay parameter."""
    return cast(_EngineNeuron, engine.ParametricLIFNeuron(a=-3.0))


def test_the_class_keeps_its_native_identity_through_pickle() -> None:
    """The class resolves through its native module for every pickle protocol."""
    constructor = engine.ParametricLIFNeuron
    assert constructor.__name__ == constructor.__qualname__ == "ParametricLIFNeuron"
    assert constructor.__module__ == _NAMESPACE
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        assert pickle.loads(pickle.dumps(constructor, protocol)) is constructor


def test_the_returned_state_is_an_independent_mapping() -> None:
    """Mutating a returned state changes neither this neuron nor another one."""
    first, other = _make(), _make()
    initial = first.get_state()
    assert tuple(initial) == ("v",)
    assert other.get_state() == initial
    returned = first.get_state()
    returned.clear()
    returned["external_key"] = 12345.0
    assert first.get_state() == initial
    assert other.get_state() == initial
    assert first.get_state() is not first.get_state()


@pytest.mark.parametrize("current", [None, "text", [], (0.125,), 0.125j])
def test_a_failed_scalar_extraction_is_atomic_and_the_neuron_recovers(current: Any) -> None:
    """A non-real input is refused with the state untouched, and the next step matches a control.

    Parameters
    ----------
    current : Any
        A value that is not a real number.
    """
    neuron, control = _make(), _make()
    neuron.step(0.75)
    control.step(0.75)
    before = neuron.get_state()
    with pytest.raises(TypeError) as refusal:
        neuron.step(current)
    assert str(refusal.value) == f"must be real number, not {type(current).__name__}"
    assert neuron.get_state() == before
    assert neuron.step(0.125) == control.step(0.125)
    assert neuron.get_state() == control.get_state()


def test_numpy_scalars_and_a_readonly_zero_rank_array_convert_like_floats() -> None:
    """NumPy scalar inputs step like their float values and leave the caller's storage alone."""
    zero_rank = np.array(0.125)
    zero_rank.setflags(write=False)
    values: list[tuple[Any, float]] = [
        (np.float16(0.125), 0.125),
        (np.float32(0.125), 0.125),
        (np.float64(0.125), 0.125),
        (np.int64(1), 1.0),
        (np.bool_(True), 1.0),
        (zero_rank, 0.125),
    ]
    for current, expected in values:
        neuron, control = _make(), _make()
        before = np.asarray(current).tobytes()
        assert neuron.step(current) == control.step(expected)
        assert neuron.get_state() == control.get_state()
        assert np.asarray(current).tobytes() == before
    assert not zero_rank.flags.writeable


def test_instances_refuse_pickling_and_copying_with_their_state_kept() -> None:
    """Instance persistence is refused for every protocol and both copiers, as for default bindings."""
    neuron = _make()
    neuron.step(0.75)
    before = neuron.get_state()
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        with pytest.raises(TypeError) as refusal:
            pickle.dumps(neuron, protocol)
        label = "ParametricLIFNeuron" if protocol < 2 else f"{_NAMESPACE}.ParametricLIFNeuron"
        assert str(refusal.value) == f"cannot pickle '{label}' object"
        assert neuron.get_state() == before
    for copier in (copy.copy, copy.deepcopy):
        with pytest.raises(TypeError) as refusal:
            copier(neuron)
        assert str(refusal.value) == f"cannot pickle '{_NAMESPACE}.ParametricLIFNeuron' object"
        assert neuron.get_state() == before


def test_steps_return_ints_and_reset_restores_a_fresh_neuron() -> None:
    """Outputs are ints that track a control, and reset returns None and equals a fresh neuron."""
    neuron, control = _make(), _make()
    for current in [-0.1, 0.0, 0.25, 0.5, 1.0, 5.0] * 4:
        output = neuron.step(current)
        assert type(output) is int
        assert output == control.step(current)
        assert neuron.get_state() == control.get_state()
    assert neuron.reset() is None
    fresh = _make()
    assert neuron.get_state() == fresh.get_state()
    assert neuron.step(0.25) == fresh.step(0.25)
    assert neuron.get_state() == fresh.get_state()
    with pytest.raises(TypeError):
        engine.ParametricLIFNeuron(current=0.25)


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

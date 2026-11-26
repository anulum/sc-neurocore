# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — HuberBraunNeuron: an explicit seed gives each neuron its own noise stream

"""A seeded HuberBraunNeuron draws its membrane noise from its own generator.

The neuron drew every noise sample from NumPy's process-wide generator, so with ``eta > 0``
its trajectory depended on everything else that drew random numbers, and two neurons could
not be given independent reproducible streams; the engine's HuberBraunNeuron has no noise at
all. With ``seed`` set the Python model owns a ``numpy.random.Generator`` seeded with it, a
refused step leaves that generator where it was, and ``reset`` restarts it. Without a seed it
still draws from ``np.random.randn``, so ``np.random.seed`` keeps governing existing code.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from sc_neurocore.neurons.models.huber_braun import HuberBraunNeuron, _safe_sigmoid

DRIVE = 5.0


def _trace(neuron: HuberBraunNeuron, steps: int = 400) -> list[float]:
    """Return the membrane potential after each of ``steps`` steps of the constant drive.

    Parameters
    ----------
    neuron : HuberBraunNeuron
        The neuron to step.
    steps : int
        Number of steps.

    Returns
    -------
    list[float]
        The potential after each step.
    """
    out = []
    for _ in range(steps):
        neuron.step(DRIVE)
        out.append(neuron.v)
    return out


def test_the_same_seed_gives_the_same_trajectory_whatever_else_draws() -> None:
    """Two neurons with one seed agree although the global generator is reseeded between them."""
    np.random.seed(1)
    first = _trace(HuberBraunNeuron(seed=7))
    np.random.seed(99)
    np.random.randn(1000)
    assert _trace(HuberBraunNeuron(seed=7)) == first


def test_a_seeded_neuron_leaves_the_global_generator_alone() -> None:
    """Stepping a seeded neuron does not advance NumPy's process-wide generator."""
    np.random.seed(5)
    expected = np.random.randn()
    np.random.seed(5)
    _trace(HuberBraunNeuron(seed=3), steps=100)
    assert np.random.randn() == expected


def test_different_seeds_give_different_trajectories() -> None:
    """Two seeds give two noise streams."""
    assert _trace(HuberBraunNeuron(seed=1)) != _trace(HuberBraunNeuron(seed=2))


def test_the_seeded_draw_is_the_generator_standard_normal() -> None:
    """The first step's potential is the Euler step with the generator's first standard-normal draw."""
    seeded = HuberBraunNeuron(seed=11)
    draw = float(np.random.default_rng(11).standard_normal())
    v = seeded.v
    arg = (v + 40.0) / 6.0
    a_sd = 0.0 + (_safe_sigmoid(arg) - 0.0) / 10.0 * 0.1
    a_sr = 0.0 + (_safe_sigmoid(-arg) - 0.0) / 20.0 * 0.1
    i_sd = 1.5 * a_sd * (v - 50.0)
    i_sr = 0.4 * a_sr * (v - -90.0)
    i_l = 0.1 * (v - -60.0)
    expected = v + (-i_sd - i_sr - i_l + DRIVE + 0.012 * draw) * 0.1
    seeded.step(DRIVE)
    assert seeded.v == expected


def test_reset_restarts_the_seeded_stream() -> None:
    """After reset a seeded neuron replays its trajectory."""
    neuron = HuberBraunNeuron(seed=42)
    first = _trace(neuron)
    neuron.reset()
    assert _trace(neuron) == first


def test_a_refused_step_leaves_the_seeded_generator_unmoved() -> None:
    """A step that overflows is refused with the generator where it was, as with every other field."""
    neuron = HuberBraunNeuron(seed=9, dt=10.0)
    control = HuberBraunNeuron(seed=9, dt=10.0)
    neuron.step(1.0)
    control.step(1.0)
    with pytest.raises(ValueError, match="non-finite"):
        neuron.step(1.7e308)
    assert neuron.step(1.0) == control.step(1.0)
    assert neuron.v == control.v


def test_an_unseeded_refused_step_keeps_the_state() -> None:
    """Without a seed an overflowing step is still refused, with the neuron's state unchanged."""
    neuron = HuberBraunNeuron(dt=10.0)
    neuron.step(1.0)
    before = (neuron.v, neuron.a_sd, neuron.a_sr)
    with pytest.raises(ValueError, match="non-finite"):
        neuron.step(1.7e308)
    assert (neuron.v, neuron.a_sd, neuron.a_sr) == before


def test_without_a_seed_the_global_generator_still_governs() -> None:
    """Unseeded neurons follow ``np.random.seed`` exactly as before."""
    np.random.seed(3)
    first = _trace(HuberBraunNeuron(), steps=200)
    np.random.seed(3)
    assert _trace(HuberBraunNeuron(), steps=200) == first


@pytest.mark.parametrize("seed", [np.int64(7), np.uint32(7)])
def test_a_numpy_integer_seed_is_the_same_seed(seed: np.integer[Any]) -> None:
    """A NumPy integer seeds the stream its Python value seeds, stored as a Python int.

    Parameters
    ----------
    seed : numpy.integer
        A NumPy integer equal to 7.
    """
    neuron = HuberBraunNeuron(seed=seed)
    assert type(neuron.seed) is int and neuron.seed == 7
    assert _trace(neuron) == _trace(HuberBraunNeuron(seed=7))


@pytest.mark.parametrize(
    ("seed", "error"),
    [
        (-1, ValueError),
        (np.int64(-1), ValueError),
        (True, TypeError),
        (np.bool_(True), TypeError),
        (1.5, TypeError),
        ("7", TypeError),
    ],
)
def test_a_bad_seed_is_refused(seed: object, error: type[Exception]) -> None:
    """A negative, boolean or non-integer seed is refused at construction.

    Parameters
    ----------
    seed : object
        The refused seed.
    error : type[Exception]
        The exception it raises.
    """
    arguments: dict[str, Any] = {"seed": seed}
    with pytest.raises(error, match="seed"):
        HuberBraunNeuron(**arguments)


@pytest.mark.parametrize("seed", [2**63, np.uint64(2**63), 2**64])
def test_a_seed_above_the_declared_domain_is_refused(seed: object) -> None:
    """A seed past ``SEED_DOMAIN``'s upper bound is refused, as one below its lower bound is.

    Parameters
    ----------
    seed : object
        An integer above ``2**63 - 1``.
    """
    arguments: dict[str, Any] = {"seed": seed}
    with pytest.raises(ValueError, match="SEED_DOMAIN"):
        HuberBraunNeuron(**arguments)


def test_both_ends_of_the_declared_domain_are_taken() -> None:
    """The lowest and highest declared seeds construct and seed a generator."""
    low, high = HuberBraunNeuron.SEED_DOMAIN
    assert (low, high) == (0, 2**63 - 1)
    assert HuberBraunNeuron(seed=low).seed == low
    assert HuberBraunNeuron(seed=high).seed == high
    assert HuberBraunNeuron(seed=np.uint64(high)).seed == high

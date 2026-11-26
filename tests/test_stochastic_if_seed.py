# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — StochasticIFNeuron: an explicit seed gives each neuron its own noise stream

"""A seeded StochasticIFNeuron draws from its own generator; an unseeded one keeps the global one.

The neuron drew every noise sample from NumPy's process-wide generator, so its spike train
depended on everything else that drew random numbers, and two neurons could not be given
independent reproducible streams. With ``seed`` set it now owns a ``numpy.random.Generator``
seeded with it, and ``reset`` restarts that stream; without a seed it still draws from
``np.random.randn``, so ``np.random.seed`` keeps governing existing code.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from sc_neurocore.neurons.models.stochastic_if import StochasticIFNeuron

DRIVE = 25.0


def _train(neuron: StochasticIFNeuron, steps: int = 500) -> list[int]:
    """Return the spike train of ``steps`` steps of the constant drive."""
    return [neuron.step(DRIVE) for _ in range(steps)]


def test_the_same_seed_gives_the_same_train_whatever_else_draws() -> None:
    """Two neurons with one seed spike identically although the global generator is reseeded between them."""
    np.random.seed(1)
    first = _train(StochasticIFNeuron(seed=7))
    np.random.seed(99)
    np.random.randn(1000)
    second = _train(StochasticIFNeuron(seed=7))
    assert first == second
    assert sum(first) > 0


def test_a_seeded_neuron_leaves_the_global_generator_alone() -> None:
    """Stepping a seeded neuron does not advance NumPy's process-wide generator."""
    np.random.seed(5)
    expected = np.random.randn()
    np.random.seed(5)
    _train(StochasticIFNeuron(seed=3), steps=200)
    assert np.random.randn() == expected


def test_different_seeds_give_different_trains() -> None:
    """Two seeds give two different noise streams."""
    assert _train(StochasticIFNeuron(seed=1)) != _train(StochasticIFNeuron(seed=2))


def test_the_seeded_draw_is_the_generator_standard_normal() -> None:
    """The first step's voltage is the Euler-Maruyama update with the generator's first draw."""
    neuron = StochasticIFNeuron(seed=11)
    draw = np.random.default_rng(11).standard_normal()
    neuron.step(0.0)
    expected = (
        -70.0 + (-(-70.0 - -70.0) + 0.0 + 0.0) / 20.0 * 1.0 + 3.0 * np.sqrt(1.0 / 20.0) * draw
    )
    assert neuron.v == expected


def test_reset_restarts_the_seeded_stream() -> None:
    """After reset a seeded neuron replays its spike train."""
    neuron = StochasticIFNeuron(seed=42)
    first = _train(neuron)
    neuron.reset()
    assert neuron.v == neuron.v_rest
    assert _train(neuron) == first


def test_without_a_seed_the_global_generator_still_governs() -> None:
    """Unseeded neurons follow ``np.random.seed`` exactly as before."""
    np.random.seed(3)
    first = _train(StochasticIFNeuron(), steps=200)
    np.random.seed(3)
    assert _train(StochasticIFNeuron(), steps=200) == first


@pytest.mark.parametrize("seed", [np.int64(7), np.int32(7), np.uint16(7), np.uint64(7)])
def test_a_numpy_integer_seed_is_the_same_seed(seed: np.integer[Any]) -> None:
    """A NumPy integer seeds the stream its Python value seeds, and is stored as a Python int."""
    neuron = StochasticIFNeuron(seed=seed)
    assert type(neuron.seed) is int and neuron.seed == 7
    assert _train(neuron) == _train(StochasticIFNeuron(seed=7))


@pytest.mark.parametrize(
    ("seed", "error"),
    [
        (-1, ValueError),
        (np.int64(-1), ValueError),
        (True, TypeError),
        (np.bool_(True), TypeError),
        (1.5, TypeError),
        (np.float64(7.0), TypeError),
        ("7", TypeError),
    ],
)
def test_a_bad_seed_is_refused(seed: object, error: type[Exception]) -> None:
    """A negative, boolean or non-integer seed is refused at construction, NumPy's included."""
    arguments: dict[str, Any] = {"seed": seed}
    with pytest.raises(error, match="seed"):
        StochasticIFNeuron(**arguments)


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
        StochasticIFNeuron(**arguments)


def test_both_ends_of_the_declared_domain_are_taken() -> None:
    """The lowest and highest declared seeds construct and seed a generator."""
    low, high = StochasticIFNeuron.SEED_DOMAIN
    assert (low, high) == (0, 2**63 - 1)
    assert StochasticIFNeuron(seed=low).seed == low
    assert StochasticIFNeuron(seed=high).seed == high
    assert StochasticIFNeuron(seed=np.uint64(high)).seed == high

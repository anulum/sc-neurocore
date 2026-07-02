# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — RNG utility contract tests

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.utils.rng import RNG


def test_rng_replays_seeded_streams_across_distributions() -> None:
    """Seeded RNG wrappers must replay equivalent streams for every drawer."""
    first = RNG(seed=42)
    second = RNG(seed=42)

    assert np.array_equal(first.random(8), second.random(8))
    assert np.array_equal(
        first.normal(mean=0.25, std=0.5, size=(2, 4)), second.normal(0.25, 0.5, (2, 4))
    )
    assert np.array_equal(first.uniform(low=-1.0, high=2.0, size=8), second.uniform(-1.0, 2.0, 8))
    assert np.array_equal(first.bernoulli(p=0.4, size=8), second.bernoulli(0.4, 8))


def test_rng_scalar_draws_return_python_scalars() -> None:
    """Scalar draws expose plain Python scalars instead of zero-dimensional arrays."""
    rng = RNG(seed=7)

    assert isinstance(RNG().random(), float)
    assert isinstance(rng.random(), float)
    assert isinstance(rng.normal(), float)
    assert isinstance(rng.uniform(), float)
    assert isinstance(rng.bernoulli(0.5), bool)


def test_rng_vector_draws_return_numpy_arrays_with_expected_dtypes() -> None:
    """Vector draws return stable NumPy array dtypes for downstream kernels."""
    rng = RNG(seed=11)

    random_samples = cast(npt.NDArray[np.float64], rng.random((2, 3)))
    normal_samples = cast(npt.NDArray[np.float64], rng.normal(size=(2, 3)))
    uniform_samples = cast(npt.NDArray[np.float64], rng.uniform(low=-3.0, high=4.0, size=(2, 3)))
    bernoulli_samples = cast(npt.NDArray[np.bool_], rng.bernoulli(p=0.25, size=(2, 3)))

    assert random_samples.shape == (2, 3)
    assert normal_samples.shape == (2, 3)
    assert uniform_samples.shape == (2, 3)
    assert bernoulli_samples.shape == (2, 3)
    assert random_samples.dtype == np.float64
    assert normal_samples.dtype == np.float64
    assert uniform_samples.dtype == np.float64
    assert bernoulli_samples.dtype == np.bool_


@pytest.mark.parametrize("seed", [-1, True])
def test_rng_rejects_invalid_seed_values(seed: int | bool) -> None:
    """Seed validation must reject values that would alias or fail downstream."""
    with pytest.raises(ValueError, match="seed"):
        RNG(seed=seed)


@pytest.mark.parametrize(
    "factory",
    [
        lambda rng: rng.normal(mean=float("nan")),
        lambda rng: rng.normal(mean=float("inf")),
        lambda rng: rng.normal(std=0.0),
        lambda rng: rng.normal(std=-1.0),
        lambda rng: rng.normal(std=float("inf")),
        lambda rng: rng.uniform(low=float("nan")),
        lambda rng: rng.uniform(high=float("inf")),
        lambda rng: rng.uniform(low=2.0, high=2.0),
        lambda rng: rng.uniform(low=3.0, high=2.0),
        lambda rng: rng.bernoulli(p=-0.1),
        lambda rng: rng.bernoulli(p=1.1),
        lambda rng: rng.bernoulli(p=float("nan")),
        lambda rng: rng.bernoulli(p=float("inf")),
    ],
)
def test_rng_rejects_invalid_distribution_parameters(factory: Callable[[RNG], object]) -> None:
    """Distribution parameters must be finite and inside documented domains."""
    with pytest.raises(ValueError):
        factory(RNG(seed=123))


def test_rng_shuffle_mutates_array_with_seeded_permutation() -> None:
    """Shuffle should mutate only the supplied NumPy array using the instance stream."""
    first = np.arange(8, dtype=np.int64)
    second = np.arange(8, dtype=np.int64)
    original: npt.NDArray[np.int64] = first.copy()

    RNG(seed=99).shuffle(first)
    RNG(seed=99).shuffle(second)

    assert np.array_equal(first, second)
    assert not np.array_equal(first, original)
    assert sorted(first.tolist()) == sorted(original.tolist())

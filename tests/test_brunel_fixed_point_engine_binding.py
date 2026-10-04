# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Fixed-point Brunel installed-engine contracts

"""Exercise distinct Brunel classes and the installed CSR simulation boundary."""

from __future__ import annotations

import pickle
from typing import Protocol, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from tests.engine_requirement import require_engine

extension = require_engine()
import sc_neurocore_engine as engine


class Network(Protocol):
    """Describe the fixed-point simulator's measured return type."""

    def run(self, n_steps: int) -> NDArray[np.uint32]:
        """Advance the network and return per-step spike counts."""
        ...


def _network() -> Network:
    """Construct four recurrent neurons through the public installed facade."""
    return cast(
        Network,
        engine.FixedPointBrunelNetwork(
            n_neurons=4,
            w_indptr=np.array([0, 1, 2, 3, 4], dtype=np.int64),
            w_indices=np.array([1, 2, 3, 0], dtype=np.int64),
            w_data=np.full(4, 128, dtype=np.int16),
            leak_k=20,
            gain_k=256,
            ext_lambda=0.7,
            ext_weight_fp=128,
            seed=17,
        ),
    )


def test_distinct_names_preserve_mean_field_behavior_and_global_identity() -> None:
    """Both classes resolve by their own names and the old population still steps."""
    population = engine.BrunelNetwork()
    before = population.get_state()
    population.step(0.5)
    assert population.get_state() != before
    population.reset()
    assert population.get_state() == before
    for name in ("BrunelNetwork", "FixedPointBrunelNetwork"):
        cls = getattr(engine, name)
        assert cls is getattr(extension, name)
        assert cls.__name__ == cls.__qualname__ == name
        assert cls.__module__ == "sc_neurocore_engine.sc_neurocore_engine"
        assert pickle.loads(pickle.dumps(cls, protocol=4)) is cls
    assert engine.BrunelNetwork is not engine.FixedPointBrunelNetwork
    assert not hasattr(engine.BrunelNetwork, "run")
    assert not hasattr(engine.FixedPointBrunelNetwork, "step")


def test_seeded_network_has_typed_counts_and_preserves_split_run_state() -> None:
    """A split run continues the same seeded recurrent and Poisson state."""
    expected = _network().run(64)
    network = _network()
    actual = np.concatenate([network.run(23), network.run(41)])
    np.testing.assert_array_equal(actual, expected)
    assert expected.dtype == np.uint32
    assert expected.shape == (64,)
    assert expected.flags.c_contiguous
    assert int(expected.sum()) > 0
    assert np.all(expected <= 4)


def test_empty_run_and_conversion_refusal_do_not_advance_state() -> None:
    """Empty execution and a negative step count leave the next trace unchanged."""
    network = _network()
    empty = network.run(0)
    assert empty.shape == (0,)
    assert empty.dtype == np.uint32
    with pytest.raises(OverflowError, match="negative"):
        network.run(-1)
    np.testing.assert_array_equal(network.run(32), _network().run(32))


def test_readonly_connectivity_is_copied_before_caller_mutation() -> None:
    """The constructor owns CSR bytes independently of the caller's arrays."""
    indptr = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    indices = np.array([1, 2, 3, 0], dtype=np.int64)
    data = np.full(4, 128, dtype=np.int16)
    for array in (indptr, indices, data):
        array.setflags(write=False)
    network = engine.FixedPointBrunelNetwork(4, indptr, indices, data, 20, 256, 0.7, 128, seed=17)
    for array in (indptr, indices, data):
        array.setflags(write=True)
        array.fill(0)
    np.testing.assert_array_equal(network.run(32), _network().run(32))


def test_malformed_csr_lengths_are_rejected_at_construction() -> None:
    """Malformed connectivity lengths raise the native constructor's ValueError."""
    with pytest.raises(ValueError, match="w_row_offsets length 2 != n_neurons\\+1=5"):
        engine.FixedPointBrunelNetwork(
            4,
            np.array([0, 1], dtype=np.int64),
            np.array([0], dtype=np.int64),
            np.array([128], dtype=np.int16),
            20,
            256,
            0.7,
            128,
        )


def test_instance_pickle_refusal_preserves_the_following_trace() -> None:
    """Unsupported instance serialization refuses without consuming network state."""
    network = _network()
    with pytest.raises(TypeError, match="cannot pickle.*FixedPointBrunelNetwork"):
        pickle.dumps(network, protocol=4)
    np.testing.assert_array_equal(network.run(32), _network().run(32))


def test_small_mean_retains_the_qualified_seeded_counts() -> None:
    """Validation and wide-current handling preserve the enrolled Knuth trace."""
    np.testing.assert_array_equal(
        _network().run(64),
        [
            1,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
            2,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            1,
            0,
            1,
            2,
            0,
            0,
            1,
            0,
            2,
            2,
            0,
            0,
            0,
            2,
            2,
            0,
            0,
            0,
            2,
            1,
            1,
            0,
            1,
            0,
            2,
            1,
            0,
            1,
            2,
            0,
            1,
            3,
            0,
            0,
            0,
            0,
            2,
            2,
            0,
            1,
            0,
            1,
            0,
        ],
    )


def test_large_mean_avoids_exponential_underflow_in_the_drive() -> None:
    """The seeded large-mean sampler drives the threshold on every measured step."""
    network = engine.FixedPointBrunelNetwork(
        1,
        np.array([0, 0], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int16),
        0,
        1,
        1000.0,
        1,
        fraction=0,
        v_threshold=900,
        refractory_period=0,
        seed=17,
    )
    np.testing.assert_array_equal(network.run(8), np.ones(8, dtype=np.uint32))


def test_duplicate_connections_wrap_large_synaptic_sums() -> None:
    """Large repeated CSR currents retain the fixed-point modular result."""
    count = 65539
    network = engine.FixedPointBrunelNetwork(
        1,
        np.array([0, count], dtype=np.int64),
        np.zeros(count, dtype=np.int64),
        np.full(count, 32767, dtype=np.int16),
        0,
        1,
        0.0,
        0,
        fraction=0,
        v_rest=1,
        v_threshold=1,
        refractory_period=0,
    )
    np.testing.assert_array_equal(network.run(2), [1, 1])


def test_empty_network_with_zero_drive_returns_owned_zero_counts() -> None:
    """An empty valid CSR network advances and returns a typed zero trace."""
    network = engine.FixedPointBrunelNetwork(
        0,
        np.array([0], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int16),
        20,
        256,
        0.0,
        128,
    )
    counts = network.run(4)
    np.testing.assert_array_equal(counts, np.zeros(4, dtype=np.uint32))
    assert counts.dtype == np.uint32
    assert counts.flags.c_contiguous
    counts.fill(1)
    next_counts = network.run(4)
    np.testing.assert_array_equal(next_counts, np.zeros(4, dtype=np.uint32))
    assert not np.shares_memory(counts, next_counts)

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Integer SHD reference contract tests

"""The integer SHD reference: input contract, delays, membrane rounding, bounds and ties."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import shd_int_reference as ref  # noqa: E402

Z1 = np.zeros(ref.N_INPUT, dtype=np.int64)
Z2 = np.zeros(ref.N_HIDDEN, dtype=np.int64)


def _network(rng: np.random.Generator, **overrides) -> ref.IntNetwork:
    fields = dict(
        frac_bits=2,
        d1=rng.integers(0, 31, ref.N_INPUT),
        d2=rng.integers(0, 31, ref.N_HIDDEN),
        q1=rng.integers(-127, 128, (ref.N_HIDDEN, ref.N_INPUT)),
        q2=rng.integers(-127, 128, (ref.N_HIDDEN, ref.N_HIDDEN)),
        q3=rng.integers(-127, 128, (ref.N_OUTPUT, ref.N_HIDDEN)),
        theta1=600,
        theta2=300,
    )
    fields.update(overrides)
    return ref.IntNetwork(**fields)


@pytest.mark.parametrize(
    "frames",
    [
        np.zeros((0, 140)),
        np.zeros((1025, 140)),
        np.zeros((4, 139)),
        -np.ones((4, 140)),
        np.full((4, 140), 0.5),
    ],
)
def test_input_outside_the_contract_is_refused(frames: np.ndarray) -> None:
    """Empty, over-long, wrongly shaped, negative and fractional inputs are refused."""
    with pytest.raises(ref.InputRefused):
        ref.run(_network(np.random.default_rng(0)), frames)


def test_counts_above_fifteen_clamp_to_fifteen() -> None:
    """Counts above fifteen give the same output as counts of fifteen."""
    net = _network(np.random.default_rng(1))
    assert np.array_equal(ref.run(net, np.full((6, 140), 40)), ref.run(net, np.full((6, 140), 15)))


def test_delay_moves_each_channel_by_exactly_its_delay() -> None:
    """Each channel is shifted by its own delay and truncated at the output length."""
    x = np.zeros((5, 3), dtype=np.int64)
    x[0] = [1, 2, 3]
    x[4] = [4, 5, 6]
    y = ref._delay(x, np.array([0, 2, 30]), 20)
    assert y[0, 0] == 1 and y[4, 0] == 4
    assert y[2, 1] == 2 and y[6, 1] == 5 and y[:2, 1].sum() == 0
    assert y[:, 2].sum() == 0  # delay 30 lands beyond the 20 output steps


def test_membrane_leak_rounds_toward_minus_infinity_on_negative_values() -> None:
    """The leak of a negative membrane rounds toward minus infinity."""
    # v = -5 after step 0. Step 1 adds 10: floor rounding gives -5 - floor(-5/4) + 10 = 7,
    # rounding toward zero would give -5 - (-1) + 10 = 6. Threshold 7 separates the two.
    acc = np.array([[-5], [10]], dtype=np.int64)
    assert ref._lif(acc, 0, theta=7, bounds=(-100, 100))[:, 0].tolist() == [0, 1]
    assert ref._lif(acc, 0, theta=8, bounds=(-100, 100))[:, 0].tolist() == [0, 0]


def test_lif_fires_at_threshold_and_resets_to_zero() -> None:
    """A neuron fires when the membrane reaches the threshold and resets to zero."""
    acc = np.array([[5], [0], [4], [0]], dtype=np.int64)
    spikes = ref._lif(acc, 0, theta=5, bounds=(-100, 100))
    # step0 u=5 fires; step1 u=0; step2 u=4; step3 u=4-1=3
    assert spikes[:, 0].tolist() == [1, 0, 0, 0]


@pytest.mark.parametrize(
    "q1_value, q3_value, theta", [(-127, 127, 1), (127, -127, 1), (127, 127, 2**30)]
)
def test_proven_bounds_hold_at_the_extremes(q1_value: int, q3_value: int, theta: int) -> None:
    """Extreme weights and thresholds on the longest full input stay within the output bound."""
    net = ref.IntNetwork(
        frac_bits=8,
        d1=Z1,
        d2=Z2,
        q1=np.full((128, 140), q1_value),
        q2=np.full((128, 128), q1_value),
        q3=np.full((20, 128), q3_value),
        theta1=theta,
        theta2=theta,
    )
    out = ref.run(net, np.full((ref.T_MAX, 140), 15))
    assert np.abs(out).max() <= ref.OUTPUT_BOUND


def test_membrane_width_covers_both_bounds() -> None:
    """The derived membrane width holds both the lowest and the highest membrane value."""
    net = _network(np.random.default_rng(2), frac_bits=4, theta1=820, theta2=636)
    for layer in (1, 2):
        low, high = net.membrane_bound(layer)
        width = net.membrane_width(layer)
        assert -(1 << (width - 1)) <= low and high < (1 << (width - 1))


def test_bounds_constants_match_the_written_arithmetic() -> None:
    """The bound constants equal their written arithmetic and fit the stated widths."""
    assert ref.L1_ACC_BOUND == 140 * 15 * 127 == 266_700 < 2**19
    assert ref.HIDDEN_ACC_BOUND == 128 * 127 == 16_256 < 2**14
    assert ref.OUTPUT_BOUND == (1024 + 30) * 16_256 == 17_133_824 < 2**25


@pytest.mark.parametrize(
    "field, value",
    [
        ("d1", np.full(140, 31)),
        ("q2", np.full((128, 128), 128)),
        ("q3", np.zeros((20, 127), dtype=np.int64)),
        ("theta1", 0),
    ],
)
def test_invalid_network_is_rejected(field: str, value) -> None:
    """Out-of-range delays, weights, shapes and thresholds are rejected."""
    with pytest.raises(ValueError):
        _network(np.random.default_rng(3), **{field: value})


def test_json_round_trip_is_exact() -> None:
    """A network survives a JSON round trip unchanged."""
    net = _network(np.random.default_rng(4))
    assert ref.IntNetwork.from_json(net.to_json()).to_json() == net.to_json()


def test_argmax_ties_go_to_the_lowest_index() -> None:
    """A tie between class accumulators predicts the lowest class index."""
    rng = np.random.default_rng(5)
    net = _network(rng, q3=np.tile(rng.integers(-127, 128, (1, 128)), (20, 1)))
    out = ref.run(net, rng.integers(0, 16, (60, 140)))
    assert len(set(out.tolist())) == 1
    assert ref.predict(net, rng.integers(0, 16, (60, 140))) == 0

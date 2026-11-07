# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bit-exact cosimulation of the integer SHD core

"""``hdl/sc_shd_int_core.v`` equals ``tools/shd_int_reference.py`` on every accumulator.

Each case compares all 20 output accumulators of every sample, in both ZERO_SKIP modes, and
checks that a refused length raises the error flag. Icarus Verilog is required: its absence
fails the test rather than skipping it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import cosim_shd_int_core as cosim  # noqa: E402
import shd_int_reference as ref  # noqa: E402

# Generous per-step deadline for the build and for one simulation of the samples below.
SIM_TIMEOUT_S = 900.0
Z1 = np.zeros(ref.N_INPUT, dtype=np.int64)
Z2 = np.zeros(ref.N_HIDDEN, dtype=np.int64)


def _random(seed: int, **overrides) -> ref.IntNetwork:
    rng = np.random.default_rng(seed)
    fields = dict(
        frac_bits=int(rng.integers(0, 5)),
        d1=rng.integers(0, 31, ref.N_INPUT),
        d2=rng.integers(0, 31, ref.N_HIDDEN),
        q1=rng.integers(-127, 128, (128, 140)),
        q2=rng.integers(-127, 128, (128, 128)),
        q3=rng.integers(-127, 128, (20, 128)),
        theta1=600,
        theta2=300,
    )
    fields.update(overrides)
    return ref.IntNetwork(**fields)


def _stimuli(seed: int) -> list:
    rng = np.random.default_rng(seed)
    impulse = np.zeros((3, 140), dtype=np.int64)
    impulse[0] = 15
    return [
        rng.integers(0, 4, (40, 140)),  # long and active ...
        rng.integers(0, 16, (5, 140)),  # ... then short: earlier slots must read as padding
        0,  # refused
        1025,  # refused
        impulse,
        np.zeros((4, 140), dtype=np.int64),
        rng.integers(0, 2, (1, 140)),  # T = 1
    ]


def _check(net: ref.IntNetwork, samples: list) -> None:
    runs = {
        zs: cosim.simulate(net, samples, timeout_seconds=SIM_TIMEOUT_S, zero_skip=zs)
        for zs in (1, 0)
    }
    for zs, results in runs.items():
        for index, (sample, result) in enumerate(zip(samples, results)):
            if isinstance(sample, int):
                assert result.refused, f"ZERO_SKIP={zs}: length {sample} was not refused"
            else:
                assert result.matches, (
                    f"ZERO_SKIP={zs} sample {index}: {result.rtl} != {result.reference}"
                )
    for event, scan in zip(runs[1], runs[0]):
        if event.rtl is not None:
            assert event.cycles == scan.cycles
            assert event.row_reads <= scan.row_reads


@pytest.mark.parametrize("seed", [0, 5])
def test_random_networks_are_bit_exact(seed: int) -> None:
    _check(_random(seed), _stimuli(seed))


@pytest.mark.parametrize(
    "name, overrides",
    [
        ("delay_0", dict(d1=Z1, d2=Z2)),
        ("delay_30", dict(d1=Z1 + 30, d2=Z2 + 30)),
        ("every_delay", dict(d1=np.arange(140) % 31, d2=np.arange(128) % 31)),
        (
            "membrane_low",
            dict(
                frac_bits=3,
                d1=Z1,
                d2=Z2,
                q1=np.full((128, 140), -127),
                q2=np.full((128, 128), -127),
                q3=np.full((20, 128), 127),
                theta1=1,
                theta2=1,
            ),
        ),
        (
            "all_fire",
            dict(
                frac_bits=3,
                d1=Z1,
                d2=Z2,
                q1=np.full((128, 140), 127),
                q2=np.full((128, 128), 127),
                q3=np.full((20, 128), -127),
                theta1=1,
                theta2=1,
            ),
        ),
        (
            "never_fire",
            dict(
                frac_bits=3,
                q1=np.full((128, 140), 127),
                q2=np.full((128, 128), 127),
                theta1=2**30,
                theta2=2**20,
            ),
        ),
    ],
)
def test_edge_networks_are_bit_exact(name: str, overrides: dict) -> None:
    _check(_random(11, **overrides), _stimuli(11))


def _leak_probe(layer: int) -> tuple[ref.IntNetwork, np.ndarray]:
    """A network whose output depends only on how the leak rounds a negative membrane.

    The probed layer sees -5 at step 0 and +10 at step 1 with threshold 7: rounding the leak
    toward minus infinity gives -5 - floor(-5 / 4) + 10 = 7 (fires), rounding toward zero
    gives -5 - (-1) + 10 = 6 (silent). Every neuron of the probed layer behaves alike, so the
    class-0 accumulator is 128 or 0.
    """
    q1 = np.zeros((128, 140), dtype=np.int64)
    q2 = np.zeros((128, 128), dtype=np.int64)
    q3 = np.zeros((20, 128), dtype=np.int64)
    q3[0, :] = 1
    if layer == 1:
        q1[:, 0], q1[:, 1] = -5, 10
        q2[np.arange(128), np.arange(128)] = 127
        theta1, theta2 = 7, 1
    else:
        q1[0, 0] = q1[1, 1] = 127
        q2[:, 0], q2[:, 1] = -5, 10
        theta1, theta2 = 1, 7
    frames = np.zeros((2, 140), dtype=np.int64)
    frames[0, 0] = frames[1, 1] = 1
    net = ref.IntNetwork(
        frac_bits=0, d1=Z1, d2=Z2, q1=q1, q2=q2, q3=q3, theta1=theta1, theta2=theta2
    )
    return net, frames


@pytest.mark.parametrize("layer", [1, 2])
def test_negative_membrane_leak_rounds_toward_minus_infinity(layer: int) -> None:
    net, frames = _leak_probe(layer)
    assert ref.run(net, frames)[0] == 128  # the probe separates the two roundings
    for zero_skip in (1, 0):
        (result,) = cosim.simulate(
            net, [frames], timeout_seconds=SIM_TIMEOUT_S, zero_skip=zero_skip
        )
        assert result.matches, (
            f"layer {layer} ZERO_SKIP={zero_skip}: {result.rtl} != {result.reference}"
        )


def test_longest_accepted_input_is_bit_exact() -> None:
    net = _random(
        13, frac_bits=8, d1=Z1, d2=Z2, q1=np.full((128, 140), 127), theta1=4000, theta2=300
    )
    results = cosim.simulate(net, [np.full((ref.T_MAX, 140), 15)], timeout_seconds=SIM_TIMEOUT_S)
    assert results[0].matches


def test_stimulus_writer_rejects_counts_outside_four_bits(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        cosim.write_stimulus([np.full((2, 140), 16)], tmp_path / "s.hex")


def test_simulate_requires_a_positive_deadline() -> None:
    with pytest.raises(ValueError):
        cosim.simulate(_random(0), [np.zeros((1, 140), dtype=np.int64)], timeout_seconds=0)

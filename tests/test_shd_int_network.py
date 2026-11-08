# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Trained integer SHD network artefact and its core run

"""The committed integer SHD network: identity, structure, and bit-exact runs on the core.

Inputs are deterministic synthetic count frames, so the test needs no dataset. Icarus Verilog
is required for the core run; its absence fails the test rather than skipping it.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import cosim_shd_int_core as cosim  # noqa: E402
import shd_int_reference as ref  # noqa: E402

NETWORK = ROOT / "hdl" / "fixtures" / "shd_int_network.json"
# sha256 of the exported JSON, which the file holds followed by one newline.
NETWORK_SHA256 = "6bb6f1387253ea5db78358d9d7076c71424e8668b9cd65e70ad14b189b0bf450"
SIM_TIMEOUT_S = 900.0


def _net() -> ref.IntNetwork:
    return ref.IntNetwork.from_json(NETWORK.read_text())


def _frames(seed: int, steps: int, high: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, high, (steps, ref.N_INPUT))


def test_network_file_is_the_exported_artefact() -> None:
    """The committed file is the exported network followed by exactly one newline."""
    data = NETWORK.read_bytes()
    assert data.endswith(b"}\n") and data.count(b"\n") == 1
    assert hashlib.sha256(data[:-1]).hexdigest() == NETWORK_SHA256


def test_network_parameters_and_bounds() -> None:
    """The network's fraction bits, thresholds, widths, delays and weight counts are pinned."""
    net = _net()
    assert (net.frac_bits, net.theta1, net.theta2) == (0, 72, 51)
    assert (net.membrane_width(1), net.membrane_width(2)) == (22, 17)
    assert net.d1.min() >= 0 and net.d2.max() <= ref.MAX_DELAY
    assert [int((q != 0).sum()) for q in (net.q1, net.q2, net.q3)] == [13652, 12265, 1638]


def test_reference_runs_the_network_on_every_input_shape() -> None:
    """The reference runs empty, typical and longest full-count inputs within the output bound."""
    net = _net()
    for frames in (
        np.zeros((1, 140), dtype=np.int64),
        _frames(1, 200, 11),
        np.full((ref.T_MAX, 140), 15),
    ):
        out = ref.run(net, frames)
        assert out.shape == (ref.N_OUTPUT,)
        assert np.abs(out).max() <= ref.OUTPUT_BOUND


def test_core_runs_the_trained_network_bit_exactly() -> None:
    """The core runs the trained network bit-exactly and refuses an over-long input."""
    net = _net()
    samples = [_frames(2, 60, 11), _frames(3, 9, 16), 1025, np.zeros((3, 140), dtype=np.int64)]
    for zero_skip in (1, 0):
        results = cosim.simulate(net, samples, timeout_seconds=SIM_TIMEOUT_S, zero_skip=zero_skip)
        for index, (sample, result) in enumerate(zip(samples, results)):
            if isinstance(sample, int):
                assert result.refused
            else:
                assert result.matches, (
                    f"ZERO_SKIP={zero_skip} sample {index}: {result.rtl} != {result.reference}"
                )
                assert any(result.reference) or not sample.any()


@pytest.mark.parametrize("frames_seed", [4])
def test_trained_network_is_not_trivial_on_active_input(frames_seed: int) -> None:
    """An active input moves the class accumulators apart."""
    # Guards the fixture: an active input must move the class accumulators.
    out = ref.run(_net(), _frames(frames_seed, 120, 11))
    assert len(set(out.tolist())) > 1

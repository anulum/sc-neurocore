# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Parameterized DCLS arithmetic against actual RTL

"""Verify declared fixed-point formats with Icarus and an unbounded integer reference."""

from __future__ import annotations

from pathlib import Path
import random
import shutil
import subprocess
from typing import TypeAlias

import pytest

_IVERILOG = shutil.which("iverilog")
_VVP = shutil.which("vvp")
_KERNEL = Path(__file__).resolve().parents[1] / "hdl" / "sc_dcls_tent_kernel.v"
pytestmark = pytest.mark.skipif(
    _IVERILOG is None or _VVP is None, reason="Icarus Verilog and vvp are required"
)
Frame: TypeAlias = tuple[int, tuple[int, ...], int, int]
Result: TypeAlias = tuple[int, int, int, int]


def _signed(raw: int, width: int) -> int:
    """Decode one raw two's-complement word without floating arithmetic."""
    return raw - (1 << width) if raw & (1 << (width - 1)) else raw


def _reference(frame: Frame, width: int, fraction: int) -> Result:
    """Compute the complete mathematical tent sum before either output saturation.

    Parameters
    ----------
    frame : tuple
        Packed spikes, signed raw tap weights, signed centre and signed sigma.
    width : int
        Signed weight and output word width.
    fraction : int
        Binary point of weights, centre, sigma and weighted output.

    Returns
    -------
    tuple
        Signed output, signed 32-bit accumulator, overflow and invalid-sigma flag.
    """
    spikes, weights, centre, sigma = frame
    total = 0
    if sigma > 0:
        for tap, weight in enumerate(weights):
            distance = abs((tap << fraction) - centre)
            if spikes & (1 << tap) and distance < sigma:
                gate = min(1 << fraction, ((sigma - distance) << fraction) // sigma)
                total += weight * gate
    word_min = -(1 << (width - 1))
    word_max = (1 << (width - 1)) - 1
    output = min(word_max, max(word_min, total >> fraction))
    accumulator = min((1 << 31) - 1, max(-(1 << 31), total))
    overflow = (
        accumulator != total or total > (word_max << fraction) or total < (word_min << fraction)
    )
    return output, accumulator, int(overflow), int(sigma <= 0)


def _simulate(tmp_path: Path, width: int, fraction: int, frames: list[Frame]) -> list[Result]:
    """Compile the public parameterized RTL and score its actual four output ports.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated simulation output directory.
    width : int
        DATA_WIDTH passed to the real kernel.
    fraction : int
        FRACTION passed to the real kernel.
    frames : list
        Independently declared raw signed input vectors.

    Returns
    -------
    list
        Actual output/accumulator/overflow/invalid-sigma observations.
    """
    assert _IVERILOG is not None and _VVP is not None
    count = len(frames[0][1])
    mask = (1 << width) - 1
    body = f"""module tb;
reg [{count - 1}:0] spikes;
reg signed [{count * width - 1}:0] weights;
reg signed [{width - 1}:0] centre, sigma;
wire signed [{width - 1}:0] result;
wire signed [31:0] accumulator;
wire overflow, invalid_sigma;
sc_dcls_tent_kernel #(.N_TAPS({count}), .DATA_WIDTH({width}), .FRACTION({fraction})) dut(
  .tap_spikes(spikes), .tap_weights_q88(weights), .centre_q88(centre), .sigma_q88(sigma),
  .weighted_sum_q88(result), .accumulator_q16_16(accumulator),
  .overflow(overflow), .invalid_sigma(invalid_sigma));
initial begin
"""
    for spikes, weights, centre, sigma in frames:
        packed = sum((word & mask) << (width * index) for index, word in enumerate(weights))
        body += (
            f"spikes={count}'h{spikes:x}; weights={count * width}'h{packed:x}; "
            f"centre={width}'h{centre & mask:x}; sigma={width}'h{sigma & mask:x}; "
            '#1; $display("OBS %h %h %b %b",result,accumulator,overflow,invalid_sigma);\n'
        )
    body += "$finish; end endmodule\n"
    source = tmp_path / "testbench.v"
    source.write_text(body, encoding="utf-8")
    executable = tmp_path / "kernel.vvp"
    subprocess.run(
        [_IVERILOG, "-g2012", "-s", "tb", "-o", str(executable), str(source), str(_KERNEL)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    run = subprocess.run(
        [_VVP, str(executable)], check=True, capture_output=True, text=True, timeout=30
    )
    observed: list[Result] = []
    for line in run.stdout.splitlines():
        if line.startswith("OBS "):
            fields = line.split()
            observed.append(
                (
                    _signed(int(fields[1], 16), width),
                    _signed(int(fields[2], 16), 32),
                    int(fields[3]),
                    int(fields[4]),
                )
            )
    assert len(observed) == len(frames), run.stdout
    return observed


@pytest.mark.parametrize(
    ("width", "fraction", "weights", "sigma", "expected"),
    [
        (12, 4, (2047, 2047), 2047, 2047),
        (12, 4, (-2048, -2048), 2047, -2048),
        (32, 16, (65536,), 65536, 65536),
        (64, 63, (1 << 62,), 1 << 62, 1 << 62),
        (16, 8, (32767, 32767), 32767, 32767),
    ],
)
def test_declared_format_has_the_correct_saturation_and_gate_precision(
    tmp_path: Path,
    width: int,
    fraction: int,
    weights: tuple[int, ...],
    sigma: int,
    expected: int,
) -> None:
    """Hand-computed cases reject hardcoded 16-bit clamps and overflowing gate numerators."""
    frame: Frame = ((1 << len(weights)) - 1, weights, 0, sigma)
    observed = _simulate(tmp_path, width, fraction, [frame])
    assert observed == [_reference(frame, width, fraction)]
    assert observed[0][0] == expected


@pytest.mark.parametrize(("width", "fraction"), [(2, 0), (12, 4), (16, 8), (32, 16), (64, 63)])
def test_live_inputs_match_full_precision_integer_contraction(
    tmp_path: Path, width: int, fraction: int
) -> None:
    """Exercise signed weights, off-centre gates, invalid sigma and both saturation directions."""
    rng = random.Random(618 + width)
    word_min = -(1 << (width - 1))
    word_max = (1 << (width - 1)) - 1
    unit = min(1 << fraction, word_max)
    frames: list[Frame] = [
        (7, (word_max, word_max, word_max), 0, word_max),
        (7, (word_min, word_min, word_min), 0, word_max),
        (0, (word_max, word_min, 0), 0, word_max),
        (7, (word_max, word_min, 0), 0, 0),
        (7, (word_max, word_min, 0), 0, -1),
    ]
    for index in range(40):
        weights = tuple(rng.randint(word_min, word_max) for _ in range(3))
        centre = (word_min, 0, word_max)[index % 3]
        sigma = (1, unit, word_max, 0, -1)[index % 5]
        frames.append((rng.randrange(8), weights, centre, sigma))
    assert _simulate(tmp_path, width, fraction, frames) == [
        _reference(frame, width, fraction) for frame in frames
    ]


def test_format_without_a_positive_radius_is_refused(tmp_path: Path) -> None:
    """A signed one-bit sigma has no positive value and is an invalid kernel format."""
    with pytest.raises(subprocess.CalledProcessError) as failure:
        _simulate(tmp_path, 1, 0, [(1, (-1,), 0, 0)])
    assert "2..64-bit data" in failure.value.stdout

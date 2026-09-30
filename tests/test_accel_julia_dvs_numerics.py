# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native Julia DVS recording acceptance

"""Exercise Julia half and extended precision through the actual file reader."""

import os
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.accel.dvs_recordings import read_dvs_recording
from tests.test_accel_julia_dvs_literals import recording
from tests.julia_runtimes import require_julia_runtime


@pytest.mark.parametrize("channel", ["1.11", "release"])
def test_julia_exhaustive_half_and_extended_rounding(tmp_path: Path, channel: str) -> None:
    """All half patterns and x87 edge representations retain exact reference float64 values."""
    paths: list[Path] = []
    half = np.arange(65536, dtype=np.uint16).view(np.float16).reshape((-1, 4))
    for byteorder in ("<", ">"):
        path = tmp_path / f"half-{byteorder == '>'}.npy"
        np.save(path, half.astype(f"{byteorder}f2"))
        paths.append(path)
        scalars = [
            (0, 1),
            (1, 16383),
            (0, 16383),
            (1 << 63, 0),
            (1 << 63, 32767),
            (0, 0),
            (0, 32768),
        ]
        payload = b""
        for significand, exponent in scalars:
            scalar = struct.pack("<QH", significand, exponent) + b"\0" * 6
            if byteorder == ">":
                scalar = scalar[::-1]
            payload += scalar * 4
        header = f"{{'descr':'{byteorder}f16','fortran_order':False,'shape':(7,4)}}\n"
        paths.append(recording(tmp_path, len(paths), header, payload))
    midpoint = np.longdouble(1) + np.longdouble(2) ** -53
    values = np.array(
        [
            [
                np.nextafter(midpoint, np.longdouble(0)),
                midpoint,
                np.nextafter(midpoint, np.longdouble(2)),
                -midpoint,
            ]
        ],
        dtype=np.longdouble,
    )
    path = tmp_path / "midpoint.npy"
    np.save(path, values)
    paths.append(path)
    for fortran in (False, True):
        header = f"{{'descr':'<f8','fortran_order':{fortran},'shape':(0,4)}}\n"
        paths.append(recording(tmp_path, len(paths), header, b""))
    api = Path(__file__).resolve().parents[1] / "src/sc_neurocore/accel/julia/datasets/dvs.jl"
    caller = tmp_path / "numeric-caller.jl"
    caller.write_text(
        "\n".join(api.read_text().splitlines()[:7])
        + "\n\n"
        + r"""
using Test
include(ARGS[1])
function main(paths)
    for path in paths
        events = @inferred DVSRecordings.read_dvs_recording(path)
        write(stdout, codeunits("DVS1"))
        write(stdout, htol(UInt64(length(events))))
        for row in axes(events,1), column in axes(events,2)
            write(stdout, htol(reinterpret(UInt64,events[row,column])))
        end
    end
end
main(ARGS[2:end])
"""
    )
    result = subprocess.run(
        [
            str(require_julia_runtime(channel)),
            "--startup-file=no",
            "--check-bounds=yes",
            "--depwarn=error",
            *(
                [f"--code-coverage={tmp_path / 'julia-numeric-%p.info'}"]
                if os.environ.get("SC_NEUROCORE_TEST_JULIA_COVERAGE") == "1"
                else []
            ),
            str(caller),
            str(api),
            *map(str, paths),
        ],
        capture_output=True,
        check=True,
        timeout=90,
    )
    offset = 0
    for path in paths:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            expected = read_dvs_recording(path, backend="numpy")
        magic, count = struct.unpack_from("<4sQ", result.stdout, offset)
        assert magic == b"DVS1" and count == expected.size
        actual = np.frombuffer(result.stdout, dtype="<f8", count=count, offset=offset + 12).reshape(
            (-1, 4)
        )
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        defined = ~np.isnan(expected)
        np.testing.assert_array_equal(
            actual[defined].view(np.uint64), expected[defined].view(np.uint64)
        )
        offset += 12 + count * 8
    assert offset == len(result.stdout)

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native Julia DVS recording acceptance

"""Exercise both installed Julia runtimes through the real recording API and CLI."""

import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.accel.dvs_recordings import read_dvs_recording


@pytest.mark.parametrize("channel", ["1.11", "release"])
def test_julia_native_real_scalar_corpus_and_inferred_return(tmp_path: Path, channel: str) -> None:
    """Every canonical scalar/version/layout combination preserves public reference bits."""
    cases: list[Path] = []
    for version in ((1, 0), (2, 0), (3, 0)):
        for fortran in (False, True):
            for descriptor in (
                "<i1",
                ">i2",
                "<i4",
                ">i8",
                "<u1",
                ">u2",
                "<u4",
                ">u8",
                "?",
                "<f2",
                ">f4",
                "<f8",
                ">f16",
            ):
                dtype = np.dtype(descriptor)
                if dtype.kind in "iu":
                    limits = np.iinfo(dtype)
                    values = np.array(
                        [[limits.min, limits.max, limits.max - 1, 0], [1, 2, 3, 4]], dtype=dtype
                    )
                elif dtype.kind == "b":
                    values = np.array(
                        [[False, True, False, True], [True, False, True, False]], dtype=dtype
                    )
                else:
                    limits_float = np.finfo(dtype)
                    values = np.array(
                        [
                            [0.0, -0.0, np.inf, -np.inf],
                            [
                                np.nan,
                                limits_float.tiny,
                                limits_float.smallest_subnormal,
                                limits_float.max,
                            ],
                        ],
                        dtype=dtype,
                    )
                path = tmp_path / f"{len(cases)}.npy"
                with path.open("wb") as stream:
                    np.lib.format.write_array(
                        stream,
                        np.array(values, order="F" if fortran else "C"),
                        version=version,
                        allow_pickle=False,
                    )
                cases.append(path)
    source = Path(__file__).resolve().parents[1]
    api = source / "src/sc_neurocore/accel/julia/datasets/dvs.jl"
    caller = tmp_path / "caller.jl"
    caller.write_text(
        "\n".join(api.read_text().splitlines()[:7])
        + "\n\n"
        + r"""
using Test
include(ARGS[1])
function main(paths)
    for path in paths
        events = @inferred DVSRecordings.read_dvs_recording(path; maximum_bytes=64)
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
            "julia",
            f"+{channel}",
            "--startup-file=no",
            "--check-bounds=yes",
            "--depwarn=error",
            str(caller),
            str(api),
            *map(str, cases),
        ],
        check=True,
        capture_output=True,
        timeout=90,
    )
    offset = 0
    for path in cases:
        magic, count = struct.unpack_from("<4sQ", result.stdout, offset)
        assert magic == b"DVS1" and count == 8
        actual = np.frombuffer(result.stdout, dtype="<f8", count=count, offset=offset + 12).reshape(
            (2, 4)
        )
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            expected = read_dvs_recording(path, backend="numpy")
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        defined = ~np.isnan(expected)
        np.testing.assert_array_equal(
            actual[defined].view(np.uint64), expected[defined].view(np.uint64)
        )
        offset += 12 + count * 8
    assert len(result.stdout) == offset


@pytest.mark.parametrize("channel", ["1.11", "release"])
@pytest.mark.parametrize("damage", ["none", "truncated", "extra", "budget", "duplicate"])
def test_julia_native_production_cli_refusals_and_frame(
    tmp_path: Path, channel: str, damage: str
) -> None:
    """The guarded production command emits exact values or refuses before binary output."""
    path = tmp_path / "events.npy"
    values = np.arange(8, dtype=np.float64).reshape((2, 4))
    np.save(path, values)
    if damage == "truncated":
        path.write_bytes(path.read_bytes()[:-1])
    elif damage == "extra":
        path.write_bytes(path.read_bytes() + b"extra")
    elif damage == "duplicate":
        header = b"{'descr':'f8','descr':'f8','shape':(2,4)}\n"
        path.write_bytes(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header)
    cli = Path(__file__).resolve().parents[1] / "src/sc_neurocore/accel/julia/datasets/dvs_cli.jl"
    result = subprocess.run(
        [
            "julia",
            f"+{channel}",
            "--startup-file=no",
            "--check-bounds=yes",
            "--depwarn=error",
            str(cli),
            str(path),
            "63" if damage == "budget" else "64",
        ],
        capture_output=True,
        timeout=30,
    )
    if damage == "none":
        assert result.returncode == 0 and result.stdout[:12] == b"DVS1\x08\0\0\0\0\0\0\0"
        np.testing.assert_array_equal(
            np.frombuffer(result.stdout[12:], dtype="<f8").reshape((2, 4)), values
        )
    else:
        assert result.returncode == 1 and result.stdout == b""
        expected_error = {
            "truncated": b"incomplete DVS recording",
            "extra": b"DVS recording has extra content",
            "budget": b"DVS event budget exceeded",
            "duplicate": b"duplicate DVS header key",
        }[damage]
        assert expected_error in result.stderr

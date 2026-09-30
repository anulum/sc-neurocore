# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native Julia DVS recording acceptance

"""Compare actual Julia public recording decisions with NumPy across inert NPY metadata."""

import os
import struct
import subprocess
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.accel.dvs_recordings import read_dvs_recording
from tests.julia_runtimes import require_julia_runtime


def recording(directory: Path, index: int, header: str, payload: bytes) -> Path:
    """Write one real UTF8 NPY recording without invoking a header evaluator."""
    path = directory / f"{index}.npy"
    raw = header.encode()
    path.write_bytes(b"\x93NUMPY\x03\x00" + struct.pack("<I", len(raw)) + raw + payload)
    return path


@pytest.mark.parametrize("channel", ["1.11", "release"])
def test_julia_header_alias_and_literal_corpus(tmp_path: Path, channel: str) -> None:
    """Native typed parser admissions and stored-value conversions match the reference file API."""
    paths: list[Path] = []
    descriptors = [
        "bool",
        "bool_",
        "?",
        "byte",
        "ubyte",
        "short",
        "ushort",
        "intc",
        "uintc",
        "long",
        "ulong",
        "longlong",
        "ulonglong",
        "int",
        "int_",
        "uint",
        "intp",
        "uintp",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "half",
        "single",
        "double",
        "float",
        "float16",
        "float32",
        "float64",
        "longdouble",
        "float128",
        "b",
        "B",
        "h",
        "H",
        "i",
        "I",
        "l",
        "L",
        "q",
        "Q",
        "p",
        "P",
        "n",
        "N",
        "e",
        "f",
        "d",
        "g",
        "u8",
        "i8",
        "f16",
        "f+8",
        "f008",
        "|f8",
        "|i8",
        ">d",
        "=int64",
        "<float64",
        "float_",
        "bool8",
        "<int",
        "<uintp",
        "<u3",
        "<f3",
        "<b2",
    ]
    descriptors.extend(
        prefix + chr(code) for prefix in ("", "<", ">", "=", "|") for code in (*range(14), 23)
    )
    for descriptor in descriptors:
        try:
            dtype = np.dtype(descriptor)
        except TypeError:
            dtype = np.dtype("<f8")
        values = np.array([[1, 2, 0, 4], [3, 4, 1, 8]], dtype=dtype)
        header = f"{{'descr':{descriptor!r},'fortran_order':False,'shape':(2,4)}}\n"
        paths.append(recording(tmp_path, len(paths), header, values.tobytes()))
    base = "{'descr': '<f8', 'fortran_order': False, 'shape': (2,4)}\n"
    headers = [
        "{'de' 'scr': '<f' '8', 'fortran_order': (False), 'shape': (+2,+4)}\n",
        "{u'descr':u'<f8',r'fortran_order':False,'shape':(0x_2,0b100)}\n",
        "{"
        + chr(39) * 3
        + "descr"
        + chr(39) * 3
        + ":"
        + chr(39) * 3
        + "<f8"
        + chr(39) * 3
        + ",'fortran_order':False,'shape':(2,4)}\n",
        "({'descr':'<f8','fortran_order':False,'shape':(+(2),+(4))})\n",
        "{'descr':'\\N{LESS-THAN SIGN}f\\N{DIGIT EIGHT}','fortran_order':False,'shape':(2,4)}\n",
        "{'\\x64escr':'<f8','fortran_order':False,'shape':(2,4)}\n",
        "{'\\u0064escr':'<f8','fortran_order':False,'shape':(2,4)}\n",
        "{'\\U00000064escr':'<f8','fortran_order':False,'shape':(2,4)}\n",
        "{'\\144escr':'<f8','fortran_order':False,'shape':(2,4)}\n",
        "{'descr':'<f8',# source scalar\n'fortran_order':False,'shape':(2,4)}\n",
        base.replace("<f8", "☃"),
        base.replace("False", "0"),
        base.replace("(2,4)", "(True,4)"),
        base.replace("(2,4)", "(2,True)"),
        base.replace("(2,4)", "(+(-2),4)"),
        base.replace("(2,4)", "(+(+2),4)"),
        base.replace("(2,4)", "(2 + 0,4)"),
        base.replace("(2,4)", "(0_0_2,4)"),
        base.replace("(2,4)", "(0x_,4)"),
        base.replace("(2,4)", "(2__0,4)"),
        base.replace("(2,4)", "(18446744073709551616,4)"),
        base.replace("(2,4)", "(" + "1" * 4301 + ",4)"),
        base.replace("}", ",'descr':'f8'}"),
        base.replace("}", ",'extra':0}"),
        " " + base,
        "\t" + base,
        "\f" + base,
        "\\\n" + base,
        "# initial comment\n" + base,
        base.rstrip() + " #\0\n",
        base.rstrip() + ";{}\n",
    ]
    for descriptor in (
        r"r'\q'",
        r"'\\'",
        r"'\''",
        r"'\q'",
        r"'\a'",
        r"'\N{LATIN CAPITAL LETTER B}'",
        r"'\N{LATIN SMALL LETTER F}8'",
        r"'\N{NOT A UNICODE NAME}'",
    ):
        headers.append("{'descr':" + descriptor + ",'fortran_order':False,'shape':(2,4)}\n")
    headers.extend(
        [
            base.replace("descr", "de\\\nscr"),
            base.replace("(2,4)", "(2,4,)"),
            base.replace("(2,4)", "(2,4,0)"),
            base.replace("(2,4)", "(2,4 0)"),
        ]
    )
    for depth in (197, 198, 199, 200, 201):
        headers.append("(" * depth + base.rstrip() + ")" * depth + "\n")
    payload = np.arange(8, dtype="<f8").tobytes()
    for header in headers:
        stored = (
            np.arange(8, dtype=np.uint8).tobytes()
            if "LATIN CAPITAL LETTER B" in header
            else payload
        )
        paths.append(recording(tmp_path, len(paths), header, stored))
    expected: list[npt.NDArray[np.float64] | None] = []
    for path in paths:
        try:
            expected.append(read_dvs_recording(path, maximum_bytes=64, backend="numpy"))
        except ValueError:
            expected.append(None)
    api = Path(__file__).resolve().parents[1] / "src/sc_neurocore/accel/julia/datasets/dvs.jl"
    caller = tmp_path / "caller.jl"
    caller.write_text(
        "\n".join(api.read_text().splitlines()[:7])
        + "\n\n"
        + r"""
using Test
include(ARGS[1])
function main(paths)
    DVSRecordings.read_dvs_recording(first(paths); maximum_bytes=64)
    descriptors = length(readdir("/proc/self/fd"))
    for path in paths
        events = try
            @inferred DVSRecordings.read_dvs_recording(path; maximum_bytes=64)
        catch error
            error isa ArgumentError || rethrow()
            @assert length(readdir("/proc/self/fd")) == descriptors
            write(stdout, codeunits("DVSE"))
            continue
        end
        @assert length(readdir("/proc/self/fd")) == descriptors
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
                [f"--code-coverage={tmp_path / 'julia-literals-%p.info'}"]
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
    for path, values_expected in zip(paths, expected, strict=True):
        if values_expected is None:
            assert result.stdout[offset : offset + 4] == b"DVSE", path.name
            offset += 4
            continue
        magic, count = struct.unpack_from("<4sQ", result.stdout, offset)
        assert magic == b"DVS1" and count == values_expected.size, path.name
        actual = np.frombuffer(result.stdout, dtype="<f8", count=count, offset=offset + 12).reshape(
            (-1, 4)
        )
        np.testing.assert_array_equal(
            np.isnan(actual), np.isnan(values_expected), err_msg=path.name
        )
        defined = ~np.isnan(values_expected)
        np.testing.assert_array_equal(
            actual[defined].view(np.uint64),
            values_expected[defined].view(np.uint64),
            err_msg=path.name,
        )
        offset += 12 + count * 8
    assert offset == len(result.stdout)

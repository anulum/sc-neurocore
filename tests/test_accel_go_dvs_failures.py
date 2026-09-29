# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native Go DVS runtime and format failures

"""Exercise actual native file, parser, argument and output failures."""

import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.accel.dvs_recordings import read_dvs_recording
from tests.test_accel_go_dvs_recordings import _native, go_dvs_executable as go_dvs_executable


@pytest.mark.parametrize(
    "header",
    [
        "{}\n",
        "( )\n",
        "'descr'\n",
        "{'descr': '<f8', 'fortran_order': 0, 'shape': (0,4)}\n",
        "{'descr': 1, 'fortran_order': False, 'shape': (0,4)}\n",
        "{'descr': '<f8', 'fortran_order': False, 'shape': ()}\n",
        "{'descr': '<f8', 'fortran_order': False, 'shape': (0,)}\n",
        "{'descr': '<f8', 'fortran_order': False, 'shape': (True,4)}\n",
        "{'descr': '<f8', 'fortran_order': False, 'shape': (0,3)}\n",
        "{'descr': '<f8', 'fortran_order': False, 'shape': (-(1),4)}\n",
        "{'descr': '<f8', 'fortran_order': False, 'shape': ((0 0),4)}\n",
        "{'descr': '<f8', 'fortran_order': False, 'shape': (0,4 4)}\n",
        "{'descr': '<f8', 'fortran_order': False, 'shape': (+(),4)}\n",
        "{'descr': '<f8', 'fortran_order': False, 'shape': ((0,4}\n",
        "{'descr': '<f8', 'fortran_order': False, 'shape': (0,4)} #\x00\n",
        "{'descr': '<f8', 'fortran_order': False, 1: (0,4)}\n",
        "{'descr' '<f8', 'fortran_order': False, 'shape': (0,4)}\n",
        "{'descr': '<f8' 'fortran_order': False, 'shape': (0,4)}\n",
        "{'descr': '<f8', 'fortran_order': False, 'shape': (0,4)\n",
        "{'descr': '<f8', 'fortran_order': False, 'shape': (0,4), 'x': 1}\n",
        "{'descr': r'\\x3cf8', 'fortran_order': False, 'shape': (0,4)}\n",
        "{'descr': '\\N!', 'fortran_order': False, 'shape': (0,4)}\n",
        "{'descr': '\\N{LATIN SMALL LETTER D', 'fortran_order': False, 'shape': (0,4)}\n",
        "{'descr': '\\U00110000', 'fortran_order': False, 'shape': (0,4)}\n",
        "{'descr': '<f8\\q', 'fortran_order': False, 'shape': (0,4)}\n",
        "{'descr': '<f8\n', 'fortran_order': False, 'shape': (0,4)}\n",
        "{'descr': '\\u\n",
        "{'descr': '<f8\\\n",
        "{'descr': '<f8', 'fortran_order': False, 'shape': (0,4),\\oops}\n",
    ],
)
def test_native_malformed_header_refusal_matches_reference(
    tmp_path: Path, go_dvs_executable: Path, header: str
) -> None:
    """Malformed inert headers refuse before allocation through both actual readers."""
    raw = header.encode()
    path = tmp_path / "invalid.npy"
    path.write_bytes(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(raw)) + raw)
    with pytest.raises(ValueError):
        read_dvs_recording(path)
    result = subprocess.run(
        [str(go_dvs_executable), str(path), "0"], capture_output=True, timeout=10
    )
    assert result.returncode != 0 and result.stdout == b""


@pytest.mark.parametrize("depth", [197, 198, 199, 200, 201])
def test_native_parenthesis_limit_matches_actual_python_parser(
    tmp_path: Path, go_dvs_executable: Path, depth: int
) -> None:
    """Valid deeply grouped headers retain parity at the parser's actual nesting boundary."""
    body = "{'descr': '<f8', 'fortran_order': False, 'shape': (0,4)}"
    header = ("(" * depth + body + ")" * depth + "\n").encode()
    path = tmp_path / "nested.npy"
    path.write_bytes(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header)
    try:
        expected = read_dvs_recording(path, maximum_bytes=0)
    except ValueError:
        result = subprocess.run(
            [str(go_dvs_executable), str(path), "0"], capture_output=True, timeout=10
        )
        assert result.returncode != 0 and result.stdout == b""
    else:
        np.testing.assert_array_equal(_native(go_dvs_executable, path, 0), expected)


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"\x93NUMPY",
        b"notNPY\x01\x00",
        b"\x93NUMPY\x04\x00",
        b"\x93NUMPY\x01\x01",
        b"\x93NUMPY\x01\x00\x01",
        b"\x93NUMPY\x01\x00\x00\x00",
        b"\x93NUMPY\x01\x00\x11\x27",
        b"\x93NUMPY\x01\x00\x10\x00short",
        b"\x93NUMPY\x03\x00\x02\x00\x00\x00\xff\n",
        b"\x93NUMPY\x01\x00\x02\x00{}",
    ],
)
def test_native_preamble_header_io_damage_refuses(
    tmp_path: Path, go_dvs_executable: Path, raw: bytes
) -> None:
    """Unsupported, incomplete or malformed binary headers never yield an event frame."""
    path = tmp_path / "prefix.npy"
    path.write_bytes(raw)
    result = subprocess.run(
        [str(go_dvs_executable), str(path), "0"], capture_output=True, timeout=10
    )
    assert result.returncode != 0 and result.stdout == b""


@pytest.mark.parametrize(
    "arguments",
    [[], ["missing"], ["missing", "bad"], ["missing", "-1"], ["", "0"], ["missing", "0"]],
)
def test_native_argument_or_file_failure_has_no_success_output(
    go_dvs_executable: Path, arguments: list[str]
) -> None:
    """Argument and filesystem refusals are surfaced as nonzero command status."""
    result = subprocess.run([str(go_dvs_executable), *arguments], capture_output=True, timeout=10)
    assert result.returncode != 0 and result.stdout == b"" and result.stderr


def test_native_payload_size_overflow_refuses_before_allocation(
    tmp_path: Path, go_dvs_executable: Path
) -> None:
    """The maximum native output budget cannot overflow an extended-width source allocation."""
    header = (
        f"{{'descr': '<f16', 'fortran_order': False, 'shape': ({sys.maxsize // 32},4)}}\n".encode()
    )
    path = tmp_path / "huge.npy"
    path.write_bytes(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header)
    result = subprocess.run(
        [str(go_dvs_executable), str(path), str(sys.maxsize)], capture_output=True, timeout=10
    )
    assert result.returncode != 0 and result.stdout == b""
    assert b"source payload exceeds native size" in result.stderr


def test_native_real_output_device_failure_is_reported(
    tmp_path: Path, go_dvs_executable: Path
) -> None:
    """An actual ENOSPC output device causes command failure after successful file decoding."""
    path = tmp_path / "events.npy"
    np.save(path, np.zeros((2, 4)))
    with Path("/dev/full").open("wb", buffering=0) as destination:
        result = subprocess.run(
            [str(go_dvs_executable), str(path), "64"],
            stdout=destination,
            stderr=subprocess.PIPE,
            timeout=10,
        )
    assert result.returncode != 0 and b"no space left" in result.stderr


@pytest.mark.parametrize("byteorder", ["<", ">"])
@pytest.mark.parametrize(
    "significand,exponent",
    [
        (0, 1),
        (1, 16383),
        (0, 16383),
        (1 << 63, 0),
        (1 << 63, 32767),
        (0, 0),
        (0, 32768),
    ],
)
def test_native_raw_extended_float_encodings_match_reference(
    tmp_path: Path, go_dvs_executable: Path, byteorder: str, significand: int, exponent: int
) -> None:
    """Raw x87 unsupported encodings convert to NaN instead of silently becoming zero."""
    scalar = struct.pack("<QH", significand, exponent) + b"\0" * 6
    if byteorder == ">":
        scalar = scalar[::-1]
    header = f"{{'descr': '{byteorder}f16', 'fortran_order': False, 'shape': (1,4)}}\n".encode()
    path = tmp_path / "extended-encoding.npy"
    path.write_bytes(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + scalar * 4)
    with np.errstate(invalid="ignore", under="ignore", over="ignore"):
        expected = read_dvs_recording(path)
    actual = _native(go_dvs_executable, path)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    defined = ~np.isnan(expected)
    np.testing.assert_array_equal(
        actual[defined].view(np.uint64), expected[defined].view(np.uint64)
    )

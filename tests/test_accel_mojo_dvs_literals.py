# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native Mojo DVS recording acceptance

"""Compare native Mojo NPY literal and datatype parsing with the public reference."""

import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.accel.dvs_recordings import read_dvs_recording
from tests.test_accel_mojo_dvs import native as _native, mojo_dvs_executable as mojo_dvs_executable


@pytest.mark.parametrize(
    "header",
    [
        "{'descr': '\\N{LESS-THAN SIGN}f\\N{DIGIT EIGHT}', 'fortran_order': False, 'shape': (2, 4)}\n",
        "{'\\x64escr': '<f8', 'fortran_order': False, 'shape': (2, 4)}\n",
        "{'\\u0064escr': '<f8', 'fortran_order': False, 'shape': (2, 4)}\n",
        "{'\\U00000064escr': '<f8', 'fortran_order': False, 'shape': (2, 4)}\n",
        "{'\\144escr': '<f8', 'fortran_order': False, 'shape': (2, 4)}\n",
        "{'\\N{LATIN SMALL LETTER D}escr': '<f\\N{DIGIT EIGHT}', 'fortran_order': False, 'shape': (2, 4)}\n",
        "{'''descr''': '''<f8''', 'fortran_order': False, 'shape': (2, 4)}\n",
        "{'descr': '<f8', 'fortran_order': False, 'shape': (+(2), +(4))}\n",
        "{'descr': '<f8', 'fortran_order': False, 'shape': (+(-2), 4)}\n",
        "{'descr': '<f8', 'fortran_order': False, 'shape': (+(+2), 4)}\n",
        " {'descr': '<f8', 'fortran_order': False, 'shape': (2, 4)}\n",
        "\t{'descr': '<f8', 'fortran_order': False, 'shape': (2, 4)}\n",
        "\n  {'descr': '<f8', 'fortran_order': False, 'shape': (2, 4)}\n",
        "{'descr': '\\xQZ', 'fortran_order': False, 'shape': (2,4)}\n",
        "{'descr': '\\N{NOT A UNICODE NAME}', 'fortran_order': False, 'shape': (2,4)}\n",
        "{'descr': '<f8', 'fortran_order': False, 'shape': (2, 4)};{}\n",
        "{'descr': '<f8', 'fortran_order': False, 'shape': (2, 4),}\n",
        "{'de' 'scr': '<f8', 'fortran_order': False, 'shape': (2, 4)}\n",
        "{u'descr': u'<f8', r'fortran_order': False, 'shape': (0x2, 0b100)}\n",
        "{'descr': '<f8', 'fortran_order': (False), 'shape': (+2, +4)}\n",
        "{'descr': '<f8', 'fortran_order': False, 'shape': ((2), (4),)}\n",
        "({'descr': '<f8', 'fortran_order': False, 'shape': (2, 4)})\n",
        "{'descr': ('<f' '8'), 'fortran_order': False, 'shape': (0o2, 0x_4)}\n",
        "{\"descr\": \"<f8\", # scalar declaration\n 'fortran_order': False, 'shape': (2, 4)}\n",
        "{'descr': '<f8', 'fortran_order': False, 'shape': (0_0_2, 4)}\n",
        "{'descr': '<f8', 'fortran_order': False, 'shape': (2, True)}\n",
        "{'descr': '<f8', 'fortran_order': False, 'shape': (2, 4), 'shape': (2, 4)}\n",
        "{'descr': '<f8', 'fortran_order': False, 'shape': (--2, 4)}\n",
        "{'descr': '<f8', 'fortran_order': False, 'shape': (2 + 0, 4)}\n",
        "{'descr': '\\a', 'fortran_order': False, 'shape': (2,4)}\n",
        "{'descr': '\\b', 'fortran_order': False, 'shape': (2,4)}\n",
        "{'descr': '\\f', 'fortran_order': False, 'shape': (2,4)}\n",
        "{'descr': '\\n', 'fortran_order': False, 'shape': (2,4)}\n",
        "{'descr': '\\r', 'fortran_order': False, 'shape': (2,4)}\n",
        "{'descr': '\\t', 'fortran_order': False, 'shape': (2,4)}\n",
        "{'descr': '\\v', 'fortran_order': False, 'shape': (2,4)}\n",
        "{'descr': '\\\\', 'fortran_order': False, 'shape': (2,4)}\n",
        "{'descr': '\\\"', 'fortran_order': False, 'shape': (2,4)}\n",
        "{'descr': '\\'', 'fortran_order': False, 'shape': (2,4)}\n",
        "{'descr': r'\\'', 'fortran_order': False, 'shape': (2,4)}\n",
        "{'descr': '\\x', 'fortran_order': False, 'shape': (2,4)}\n",
        "{'descr': '\\u000G', 'fortran_order': False, 'shape': (2,4)}\n",
        "{'descr': '\\ud800', 'fortran_order': False, 'shape': (2,4)}\n",
        "{'descr': '\\N{LATIN CAPITAL LETTER F}8', 'fortran_order': False, 'shape': (2,4)}\n",
        "{'descr': '\\N{GREATER-THAN SIGN}f8', 'fortran_order': False, 'shape': (2,4)}\n",
        "{'descr': '\\N{EQUALS SIGN}f8', 'fortran_order': False, 'shape': (2,4)}\n",
        "{'descr': '\\N{VERTICAL LINE}f8', 'fortran_order': False, 'shape': (2,4)}\n",
        "{'descr': '\\N{QUESTION MARK}', 'fortran_order': False, 'shape': (2,4)}\n",
        "{'descr': 'f\\N{PLUS SIGN}8', 'fortran_order': False, 'shape': (2,4)}\n",
        "{'descr': 'f\\N{HYPHEN-MINUS}8', 'fortran_order': False, 'shape': (2,4)}\n",
        "{'descr': '\\377', 'fortran_order': False, 'shape': (2,4)}\n",
        "{'descr': '\\777', 'fortran_order': False, 'shape': (2,4)}\n",
        "{'descr': '☃', 'fortran_order': False, 'shape': (2,4)}\n",
        "{'descr': '\\é', 'fortran_order': False, 'shape': (2,4)}\n",
        "{'descr': '\\☃', 'fortran_order': False, 'shape': (2,4)}\n",
        "{'descr': '\\😀', 'fortran_order': False, 'shape': (2,4)}\n",
        "{'descr': r'\\é', 'fortran_order': False, 'shape': (2,4)}\n",
        "{'descr': r'\\☃', 'fortran_order': False, 'shape': (2,4)}\n",
        "{'descr': r'\\😀', 'fortran_order': False, 'shape': (2,4)}\n",
        "{'descr': '<u3', 'fortran_order': False, 'shape': (2,4)}\n",
        "{'descr': '<f3', 'fortran_order': False, 'shape': (2,4)}\n",
        "{'descr': '<b2', 'fortran_order': False, 'shape': (2,4)}\n",
        "{'descr': '<int', 'fortran_order': False, 'shape': (2,4)}\n",
        "{'descr': '<uintp', 'fortran_order': False, 'shape': (2,4)}\n",
        "\x0c{'descr': '<f8', 'fortran_order': False, 'shape': (2,4)}\n",
        "# initial comment\n{'descr': '<f8', 'fortran_order': False, 'shape': (2,4)}\n",
        "\\\n{'descr': '<f8', 'fortran_order': False, 'shape': (2,4)}\n",
        "{'descr': '<f8', 'fortran_order': False, 'shape': (2,4,0)}\n",
        "{'descr': '<f8', 'fortran_order': False, 'shape': (2,4 0)}\n",
        "{'descr': '<f8', 'fortran_order': False, 'shape': (0x_,4)}\n",
        "{'descr': '<f8', 'fortran_order': False, 'shape': (2__0,4)}\n",
        "{'descr': '<f8', 'fortran_order': False, 'shape': (2_,4)}\n",
        "{'descr': '<f8', 'fortran_order': False, 'shape': (0xg,4)}\n",
        "{'descr': '<f8', 'fortran_order': False, 'shape': (,4)}\n",
        "{'descr': '<f8', 'fortran_order': False, 'shape': (-2,4)}\n",
        "{'descr': '<f8', 'fortran_order': False, 'shape': (" + "1" * 4301 + ",4)}\n",
        "{'descr': '''<f8\n''', 'fortran_order': False, 'shape': (2,4)}\n",
        "{'descr': '<f\\\n8', 'fortran_order': False, 'shape': (2,4)}\n",
    ],
)
def test_mojo_native_python_literal_header_acceptance_matches_reference(
    tmp_path: Path, mojo_dvs_executable: Path, header: str
) -> None:
    """Actual literal variants and invalid expressions share the reference admission decision."""
    path = tmp_path / "literal.npy"
    raw = header.encode()
    values = np.arange(8, dtype="<f8").reshape((2, 4))
    path.write_bytes(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(raw)) + raw + values.tobytes())
    try:
        expected = read_dvs_recording(path)
    except ValueError:
        result = subprocess.run(
            [str(mojo_dvs_executable), str(path), "64"], capture_output=True, timeout=10
        )
        assert result.returncode != 0 and result.stdout == b""
    else:
        np.testing.assert_array_equal(_native(mojo_dvs_executable, path), expected)


@pytest.mark.parametrize(
    "descriptor",
    [
        *[prefix + chr(code) for prefix in ("", "<", ">", "=", "|") for code in (*range(14), 23)],
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
    ],
)
def test_mojo_native_dtype_alias_matches_reference(
    tmp_path: Path, mojo_dvs_executable: Path, descriptor: str
) -> None:
    """NumPy named/character/native-width aliases preserve real-file decoding and refusal."""
    try:
        dtype = np.dtype(descriptor)
    except TypeError:
        dtype = np.dtype("<f8")
    values = np.array([[1, 2, 0, 4], [3, 4, 1, 8]], dtype=dtype)
    header = f"{{'descr': {descriptor!r}, 'fortran_order': False, 'shape': (2, 4)}}\n".encode()
    path = tmp_path / "alias.npy"
    path.write_bytes(
        b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + values.tobytes()
    )
    try:
        expected = read_dvs_recording(path)
    except ValueError:
        result = subprocess.run(
            [str(mojo_dvs_executable), str(path), "64"], capture_output=True, timeout=10
        )
        assert result.returncode != 0 and result.stdout == b""
    else:
        np.testing.assert_array_equal(_native(mojo_dvs_executable, path), expected)

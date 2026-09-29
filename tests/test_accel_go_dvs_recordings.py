# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native Go DVS recording acceptance

"""Compare real native Go CLI recording results with the public NumPy reference."""

import struct
import subprocess
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.accel.dvs_recordings import read_dvs_recording


@pytest.fixture(scope="module")
def go_dvs_executable(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Compile the actual native recording command without cached fixture substitutions."""
    executable = tmp_path_factory.mktemp("native") / "dvs-go"
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        ["go", "build", "-o", str(executable), "./services/loaders/dvscli"],
        cwd=root / "src/sc_neurocore/accel/go",
        check=True,
        capture_output=True,
        timeout=120,
    )
    return executable


def _native(
    executable: Path, path: Path, budget: int = 64 * 1024 * 1024
) -> npt.NDArray[np.float64]:
    """Exercise the binary command and validate its complete success frame."""
    result = subprocess.run(
        [str(executable), str(path), str(budget)], capture_output=True, check=True, timeout=10
    )
    assert result.stdout[:4] == b"DVS1"
    count = struct.unpack("<Q", result.stdout[4:12])[0]
    assert len(result.stdout) == 12 + count * 8
    return np.frombuffer(result.stdout[12:], dtype="<f8").reshape((-1, 4))


@pytest.mark.parametrize("version", [(1, 0), (2, 0), (3, 0)])
@pytest.mark.parametrize("fortran", [False, True])
@pytest.mark.parametrize(
    "dtype",
    ["<i1", ">i2", "<i4", ">i8", "<u1", ">u2", "<u4", ">u8", "?", "<f2", ">f4", "<f8", ">f16"],
)
def test_go_native_numeric_values_match_numpy(
    tmp_path: Path, go_dvs_executable: Path, version: tuple[int, int], fortran: bool, dtype: str
) -> None:
    """Canonical NPY widths, byte orders and layouts preserve actual stored numeric values."""
    scalar = np.dtype(dtype)
    if scalar.kind in "iu":
        limits = np.iinfo(scalar)
        values = np.array([[limits.min, limits.max, limits.max - 1, 0], [1, 2, 3, 4]], dtype=scalar)
    elif scalar.kind == "b":
        values = np.array([[False, True, False, True], [True, False, True, False]], dtype=scalar)
    else:
        float_limits = np.finfo(scalar)
        values = np.array(
            [
                [0.0, -0.0, np.inf, -np.inf],
                [np.nan, float_limits.tiny, float_limits.smallest_subnormal, float_limits.max],
            ],
            dtype=scalar,
        )
    recorded = np.array(values, dtype=scalar, order="F" if fortran else "C")
    path = tmp_path / "events.npy"
    with path.open("wb") as stream:
        np.lib.format.write_array(stream, recorded, version=version, allow_pickle=False)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        expected = read_dvs_recording(path)
    actual = _native(go_dvs_executable, path)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    defined = ~np.isnan(expected)
    np.testing.assert_array_equal(
        actual[defined].view(np.uint64), expected[defined].view(np.uint64)
    )


@pytest.mark.parametrize(
    "damage", ["truncated", "extra", "complex", "object", "duplicate", "budget"]
)
def test_go_native_refuses_real_damaged_or_unsupported_files(
    tmp_path: Path, go_dvs_executable: Path, damage: str
) -> None:
    """Refused files never produce a successful binary event frame or a synthetic replacement."""
    path = tmp_path / "events.npy"
    dtype = "complex128" if damage == "complex" else "object" if damage == "object" else "float64"
    np.save(path, np.ones((2, 4), dtype=dtype))
    if damage == "truncated":
        path.write_bytes(path.read_bytes()[:-1])
    elif damage == "extra":
        path.write_bytes(path.read_bytes() + b"extra")
    elif damage == "duplicate":
        header = b"{'descr': '<f8', 'descr': '<f8', 'shape': (2, 4)}\n"
        path.write_bytes(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header)
    budget = 63 if damage == "budget" else 64
    result = subprocess.run(
        [str(go_dvs_executable), str(path), str(budget)], capture_output=True, timeout=10
    )
    assert result.returncode != 0
    assert result.stdout == b""


def test_go_native_extended_midpoint_rounding_matches_reference(
    tmp_path: Path, go_dvs_executable: Path
) -> None:
    """Extended significands round once to float64 across both sides of a midpoint."""
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
    np.testing.assert_array_equal(
        _native(go_dvs_executable, path).view(np.uint64), read_dvs_recording(path).view(np.uint64)
    )


@pytest.mark.parametrize("fortran", [False, True])
@pytest.mark.parametrize("rows", [0, 2])
def test_go_native_reordered_header_and_exact_budget(
    tmp_path: Path, go_dvs_executable: Path, fortran: bool, rows: int
) -> None:
    """Key order is irrelevant and empty/exact event budgets are accepted."""
    values = np.arange(rows * 4, dtype="<f8").reshape((rows, 4))
    header = (f'{{"shape": ({rows}, 4), "fortran_order": {fortran}, "descr": "<f8"}}\n').encode()
    path = tmp_path / "reordered.npy"
    path.write_bytes(
        b"\x93NUMPY\x01\x00"
        + struct.pack("<H", len(header))
        + header
        + values.tobytes(order="F" if fortran else "C")
    )
    np.testing.assert_array_equal(_native(go_dvs_executable, path, rows * 32), values)


@pytest.mark.parametrize(
    "header",
    [
        b"{'descr\": '<f8', 'fortran_order': False, 'shape': (0,4)}\n",
        b"{'descr': '<f8\", 'fortran_order': False, 'shape': (0,4)}\n",
    ],
)
def test_go_native_mismatched_python_quotes_refuse(
    tmp_path: Path, go_dvs_executable: Path, header: bytes
) -> None:
    """Malformed Python literals cannot be admitted as an empty recording."""
    path = tmp_path / "malformed.npy"
    path.write_bytes(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header)
    result = subprocess.run(
        [str(go_dvs_executable), str(path), "0"], capture_output=True, timeout=10
    )
    assert result.returncode != 0 and result.stdout == b""

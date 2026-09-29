# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native Rust DVS recording acceptance

"""Exercise the native Rust command against actual reference NPY recordings."""

import struct
import subprocess
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.accel.dvs_recordings import read_dvs_recording


@pytest.fixture(scope="module")
def rust_dvs_executable(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the production Rust command and its host C extended scalar bridge."""
    target = tmp_path_factory.mktemp("rust-dvs")
    crate = Path(__file__).resolve().parents[1] / "src/sc_neurocore/accel/rust/safety/dvs_native"
    subprocess.run(
        [
            "cargo",
            "build",
            "--offline",
            "--locked",
            "--release",
            "--manifest-path",
            str(crate / "Cargo.toml"),
            "--target-dir",
            str(target),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return target / "release/sc-neurocore-dvs"


def native(executable: Path, path: Path, budget: int = 64 * 1024 * 1024) -> npt.NDArray[np.float64]:
    """Decode the complete production transport frame without substituting the reader."""
    result = subprocess.run(
        [str(executable), str(path), str(budget)], check=True, capture_output=True, timeout=10
    )
    assert result.stdout[:4] == b"DVS1"
    count = struct.unpack("<Q", result.stdout[4:12])[0]
    assert count % 4 == 0 and len(result.stdout) == 12 + count * 8
    return np.frombuffer(result.stdout[12:], dtype="<f8").reshape((-1, 4))


@pytest.mark.parametrize("version", [(1, 0), (2, 0), (3, 0)])
@pytest.mark.parametrize("fortran", [False, True])
@pytest.mark.parametrize(
    "dtype",
    ["<i1", ">i2", "<i4", ">i8", "<u1", ">u2", "<u4", ">u8", "?", "<f2", ">f4", "<f8", ">f16"],
)
def test_rust_real_numeric_storage(
    tmp_path: Path, rust_dvs_executable: Path, version: tuple[int, int], fortran: bool, dtype: str
) -> None:
    """Integer extrema and floating special values preserve reference float64 bits."""
    scalar = np.dtype(dtype)
    if scalar.kind in "iu":
        limits = np.iinfo(scalar)
        values = np.array([[limits.min, limits.max, limits.max - 1, 0], [1, 2, 3, 4]], dtype=scalar)
    elif scalar.kind == "b":
        values = np.array([[False, True, False, True], [True, False, True, False]], dtype=scalar)
    else:
        limits_float = np.finfo(scalar)
        values = np.array(
            [
                [0.0, -0.0, np.inf, -np.inf],
                [np.nan, limits_float.tiny, limits_float.smallest_subnormal, limits_float.max],
            ],
            dtype=scalar,
        )
    path = tmp_path / "events.npy"
    with path.open("wb") as stream:
        np.lib.format.write_array(
            stream,
            np.array(values, order="F" if fortran else "C"),
            version=version,
            allow_pickle=False,
        )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        expected = read_dvs_recording(path, backend="numpy")
    actual = native(rust_dvs_executable, path, 64)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    mask = ~np.isnan(expected)
    np.testing.assert_array_equal(actual[mask].view(np.uint64), expected[mask].view(np.uint64))


@pytest.mark.parametrize(
    "shape",
    [
        "(2,4)",
        "(+(2),+(4))",
        "(0x_2,0b100)",
        "(0_0_2,4)",
        "(True,4)",
        "(2,True)",
        "(+(-2),4)",
        "(+(+2),4)",
        "(18446744073709551616,4)",
        "(2 + 0,4)",
    ],
)
@pytest.mark.parametrize(
    "descriptor",
    [
        "'<f8'",
        "'float64'",
        "'\\u003cf8'",
        "'\\144'",
        "'\\N{LESS-THAN SIGN}f8'",
        "'☃'",
        "'\\U00110000'",
        "r'\\x3cf8'",
        "'<f' '8'",
        "u'<f8'",
    ],
)
def test_rust_inert_header_reference_decisions(
    tmp_path: Path, rust_dvs_executable: Path, shape: str, descriptor: str
) -> None:
    """Admitted escaped literals and malformed expressions match public reference decisions."""
    header = f"{{'descr': {descriptor}, 'fortran_order': False, 'shape': {shape}}}\n".encode()
    path = tmp_path / "literal.npy"
    values = np.arange(8, dtype="<f8").reshape((2, 4))
    path.write_bytes(
        b"\x93NUMPY\x03\x00" + struct.pack("<I", len(header)) + header + values.tobytes()
    )
    try:
        expected = read_dvs_recording(path, maximum_bytes=64, backend="numpy")
    except ValueError:
        result = subprocess.run(
            [str(rust_dvs_executable), str(path), "64"], capture_output=True, timeout=10
        )
        assert result.returncode != 0 and result.stdout == b""
        assert b"panicked" not in result.stderr
    else:
        np.testing.assert_array_equal(native(rust_dvs_executable, path, 64), expected)


@pytest.mark.parametrize("damage", ["truncated", "extra", "complex", "object", "budget"])
def test_rust_recording_refusal(tmp_path: Path, rust_dvs_executable: Path, damage: str) -> None:
    """Invalid payloads, real nonnumeric arrays and exhausted budgets emit no success frame."""
    path = tmp_path / "events.npy"
    dtype = "complex128" if damage == "complex" else "object" if damage == "object" else "float64"
    np.save(path, np.ones((2, 4), dtype=dtype))
    if damage == "truncated":
        path.write_bytes(path.read_bytes()[:-1])
    elif damage == "extra":
        path.write_bytes(path.read_bytes() + b"extra")
    result = subprocess.run(
        [str(rust_dvs_executable), str(path), "63" if damage == "budget" else "64"],
        capture_output=True,
        timeout=10,
    )
    assert result.returncode != 0 and result.stdout == b""


@pytest.mark.parametrize("depth", [197, 198, 199, 200, 201])
def test_rust_structural_nesting_limit(
    tmp_path: Path, rust_dvs_executable: Path, depth: int
) -> None:
    """Structural depth retains actual Python parser admission at the nesting boundary."""
    text = "(" * depth + "{'descr':'<f8','fortran_order':False,'shape':(0,4)}" + ")" * depth + "\n"
    header = text.encode()
    path = tmp_path / "empty.npy"
    path.write_bytes(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header)
    try:
        expected = read_dvs_recording(path, maximum_bytes=0, backend="numpy")
    except ValueError:
        result = subprocess.run(
            [str(rust_dvs_executable), str(path), "0"], capture_output=True, timeout=10
        )
        assert result.returncode != 0 and result.stdout == b""
    else:
        np.testing.assert_array_equal(native(rust_dvs_executable, path, 0), expected)


def test_rust_extended_midpoint(tmp_path: Path, rust_dvs_executable: Path) -> None:
    """Host extended precision rounds once across both sides of a float64 midpoint."""
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
        native(rust_dvs_executable, path, 32).view(np.uint64),
        read_dvs_recording(path, backend="numpy").view(np.uint64),
    )


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["missing"],
        ["missing", "bad"],
        ["missing", "-1"],
        ["missing", "0", "0"],
        ["missing", "0", "-1"],
        ["missing", "0", "2147483647"],
        ["missing", "0", "1", "extra"],
    ],
)
def test_rust_native_argument_and_parent_refusal(
    rust_dvs_executable: Path, arguments: list[str]
) -> None:
    """Malformed native arguments and absent expected parents refuse before file access."""
    result = subprocess.run([str(rust_dvs_executable), *arguments], capture_output=True, timeout=10)
    assert result.returncode != 0 and result.stdout == b""
    assert b"panicked" not in result.stderr

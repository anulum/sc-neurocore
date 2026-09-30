# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native DVS numeric scalar conversion

"""Check the actual Mojo scalar C API against NumPy numeric conversion."""

import ctypes
from pathlib import Path
from collections.abc import Callable
from typing import cast
import subprocess

import numpy as np
import pytest
from sc_neurocore.accel.mojo.isa_baseline import pin_isa


DecodeScalar = Callable[[int, int, int, int, int, int, int], int]


@pytest.fixture(scope="module")
def native(tmp_path_factory: pytest.TempPathFactory) -> DecodeScalar:
    """Load the compiled actual Mojo C boundary with its complete ABI."""
    root = Path(__file__).resolve().parents[1]
    directory = tmp_path_factory.mktemp("mojo-dvs-scalar")
    object = directory / "extended.o"
    subprocess.run(
        [
            "cc",
            "-fPIC",
            "-c",
            str(root / "src/sc_neurocore/accel/rust/safety/dvs_native/src/extended.c"),
            "-o",
            str(object),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    path = directory / "libdvs_scalar.so"
    subprocess.run(
        pin_isa(
            [
                "mojo",
                "build",
                str(root / "src/sc_neurocore/accel/mojo/kernels/dvs_scalar.mojo"),
                "--emit",
                "shared-lib",
                "--Werror",
                "--diagnose-missing-doc-strings",
                "--fp-mode",
                "contract=off",
                "-j",
                "2",
                "-Xlinker",
                str(object),
                "-o",
                str(path),
            ]
        ),
        check=True,
        capture_output=True,
        timeout=120,
    )
    library = ctypes.CDLL(str(path))
    function = library.dvs_scalar_decode_c
    function.argtypes = [ctypes.c_ssize_t] * 7
    function.restype = ctypes.c_int32
    return cast(DecodeScalar, function)


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize(
    "dtype", ["b1", "i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f2", "f4", "f8", "f16"]
)
def test_numeric_storage_bits_match_numpy(native: DecodeScalar, endian: str, dtype: str) -> None:
    """Integer extrema, signed zero and floating boundaries widen exactly once."""
    scalar = np.dtype(endian + dtype)
    if scalar.kind in "iu":
        limits = np.iinfo(scalar)
        values = np.array([limits.min, limits.max, limits.max - 1, 0, 1], dtype=scalar)
    elif scalar.kind == "b":
        values = np.array([False, True], dtype=scalar)
    else:
        float_limits = np.finfo(scalar)
        values = np.array(
            [
                0,
                -0.0,
                np.inf,
                -np.inf,
                np.nan,
                float_limits.tiny,
                float_limits.smallest_subnormal,
                float_limits.max,
            ],
            dtype=scalar,
        )
        if scalar.itemsize == 16:
            midpoint = np.longdouble(1) + np.longdouble(2) ** -53
            values = np.concatenate(
                (
                    values,
                    np.array(
                        [
                            np.nextafter(midpoint, np.longdouble(0)),
                            midpoint,
                            np.nextafter(midpoint, np.longdouble(2)),
                            -midpoint,
                        ],
                        dtype=scalar,
                    ),
                )
            ).astype(scalar)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        expected = values.astype(np.float64)
    actual = np.empty(len(values), dtype=np.float64)
    raw = values.tobytes()
    source = ctypes.create_string_buffer(b"prefix" + raw)
    for index in range(len(values)):
        out = ctypes.c_double(-999)
        assert (
            native(
                ctypes.addressof(source),
                len(source.raw) - 1,
                6 + index * scalar.itemsize,
                ord(scalar.kind),
                scalar.itemsize,
                int(endian == ">"),
                ctypes.addressof(out),
            )
            == 0
        )
        actual[index] = out.value
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    mask = ~np.isnan(expected)
    np.testing.assert_array_equal(actual[mask].view(np.uint64), expected[mask].view(np.uint64))


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize(
    "dtype", ["i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f2", "f4", "f8"]
)
def test_random_stored_bits_match_numpy(native: DecodeScalar, endian: str, dtype: str) -> None:
    """Deterministic raw storage samples catch sign extension and endian mistakes."""
    scalar = np.dtype(endian + dtype)
    raw = np.random.default_rng(819).bytes(scalar.itemsize * 256)
    source = ctypes.create_string_buffer(raw)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        expected = np.frombuffer(raw, dtype=scalar).astype(np.float64)
    actual = np.empty(256)
    for index in range(256):
        out = ctypes.c_double(-999)
        assert (
            native(
                ctypes.addressof(source),
                len(raw),
                index * scalar.itemsize,
                ord(scalar.kind),
                scalar.itemsize,
                int(endian == ">"),
                ctypes.addressof(out),
            )
            == 0
        )
        actual[index] = out.value
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    mask = ~np.isnan(expected)
    np.testing.assert_array_equal(actual[mask].view(np.uint64), expected[mask].view(np.uint64))


@pytest.mark.parametrize(
    "damage",
    [
        "negative_size",
        "negative_offset",
        "truncated",
        "past_end",
        "invalid_kind",
        "invalid_width",
        "bad_boolean_width",
        "bad_float_width",
        "bad_endian",
        "null_source",
        "null_output",
        "unaligned_output",
        "source_overflow",
        "output_overflow",
        "overlap",
    ],
)
def test_refusal_preserves_destination(native: DecodeScalar, damage: str) -> None:
    """Invalid caller declarations refuse before dereference or output mutation."""
    source = ctypes.create_string_buffer(32)
    output = ctypes.c_double(-999)
    args = [ctypes.addressof(source), 32, 0, ord("f"), 8, 0, ctypes.addressof(output)]
    changes = {
        "negative_size": (1, -1),
        "negative_offset": (2, -1),
        "truncated": (1, 7),
        "past_end": (2, 25),
        "invalid_kind": (3, ord("c")),
        "invalid_width": (4, 3),
        "bad_endian": (5, 2),
        "null_source": (0, 0),
        "null_output": (6, 0),
        "unaligned_output": (6, ctypes.addressof(output) + 1),
        "source_overflow": (0, (1 << 63) - 16),
        "output_overflow": (6, (1 << 63) - 8),
        "overlap": (6, ctypes.addressof(source)),
    }
    if damage == "bad_boolean_width":
        args[3:5] = [ord("b"), 8]
    elif damage == "bad_float_width":
        args[4] = 1
    else:
        index, value = changes[damage]
        args[index] = value
    assert native(*args) == -1
    assert output.value == -999
    assert source.raw == b"\0" * 32

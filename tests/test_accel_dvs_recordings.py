# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Converted DVS recording acceptance

"""Exercise real NPY files through the bounded reader and both dataset entry points."""

import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sc_neurocore.accel.dvs_recordings import read_dvs_recording
from sc_neurocore.datasets import load_dvs_cifar10
from sc_neurocore.datasets.event_samples import read_event_sample
from sc_neurocore.datasets.manifest import SampleRecord


@pytest.mark.parametrize("version", [(1, 0), (2, 0), (3, 0)])
@pytest.mark.parametrize("fortran", [False, True])
@pytest.mark.parametrize("dtype", ["<f2", ">f8", ">i8", "<u4", "?", "<f16"])
def test_real_npy_versions_orders_and_numeric_types_preserve_values(
    tmp_path: Path, version: tuple[int, int], fortran: bool, dtype: str
) -> None:
    """Every supported file version/layout keeps values and yields independent writable doubles."""
    path = tmp_path / "events.npy"
    recorded = np.array(
        [[1, 2, 1, 1.002], [3, 4, 0, 2.004]], dtype=dtype, order="F" if fortran else "C"
    )
    with path.open("wb") as stream:
        np.lib.format.write_array(stream, recorded, version=version, allow_pickle=False)
    actual = read_dvs_recording(path)
    np.testing.assert_array_equal(actual, recorded.astype(np.float64))
    assert actual.dtype == np.float64 and actual.flags.c_contiguous and actual.flags.owndata
    assert actual.flags.writeable
    actual[0, 0] = 12
    np.testing.assert_array_equal(read_dvs_recording(path), recorded.astype(np.float64))


@pytest.mark.parametrize("rows,budget", [(0, 0), (2, 64)])
def test_empty_and_exact_result_budget_are_admitted(tmp_path: Path, rows: int, budget: int) -> None:
    """Empty recordings allocate no event matrix and exact byte bounds are inclusive."""
    path = tmp_path / "events.npy"
    recorded = np.zeros((rows, 4), dtype=np.float16)
    np.save(path, recorded)
    np.testing.assert_array_equal(read_dvs_recording(path, maximum_bytes=budget), recorded)


@pytest.mark.parametrize("budget", [-1, True, "64", 63])
def test_invalid_or_insufficient_budget_refuses(tmp_path: Path, budget: Any) -> None:
    """Invalid declarations or an oversized matrix fail before event decoding."""
    path = tmp_path / "events.npy"
    np.save(path, np.zeros((2, 4)))
    with pytest.raises(ValueError, match="budget"):
        read_dvs_recording(path, maximum_bytes=budget)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("dtype", ["complex128", "U8", "S8", "object"])
def test_nonreal_numerical_coercion_is_refused_through_both_public_paths(
    tmp_path: Path, lazy: bool, dtype: str
) -> None:
    """Complex, text and pickled event values never become a successful real recording."""
    path = tmp_path / "train/0/events.npy"
    path.parent.mkdir(parents=True)
    values = [[1 + 2j, 2, 1, 1.002]] if dtype == "complex128" else [[1, 2, 1, 1.002]]
    np.save(path, np.array(values, dtype=dtype))
    with pytest.raises(ValueError, match="real numeric"):
        if lazy:
            read_event_sample(
                tmp_path,
                "dvs_cifar10",
                SampleRecord("train", "train/0/events.npy", 0, 0, "recording"),
            )
        else:
            load_dvs_cifar10(tmp_path)


@pytest.mark.parametrize("damage", ["truncated", "extra", "concatenated"])
@pytest.mark.parametrize("lazy", [False, True])
def test_payload_damage_cannot_be_silently_ignored(tmp_path: Path, damage: str, lazy: bool) -> None:
    """Each file contains exactly one complete event array across eager and manifest readers."""
    path = tmp_path / "train/0/events.npy"
    path.parent.mkdir(parents=True)
    np.save(path, np.zeros((2, 4)))
    if damage == "truncated":
        path.write_bytes(path.read_bytes()[:-1])
    elif damage == "extra":
        with path.open("ab") as stream:
            stream.write(b"unbound extra content")
    else:
        with path.open("ab") as stream:
            np.save(stream, np.zeros((2, 4)))
    with pytest.raises(ValueError, match="incomplete|extra content"):
        if lazy:
            read_event_sample(
                tmp_path,
                "dvs_cifar10",
                SampleRecord("train", "train/0/events.npy", 0, 0, "recording"),
            )
        else:
            load_dvs_cifar10(tmp_path)


@pytest.mark.parametrize(
    "header",
    [
        b"[]\n",
        b"{\n",
        b"{[]: '<f8', 'fortran_order': False, 'shape': (1, 4)}\n",
        b"{'descr': '<f8', 'descr': '<i8', 'shape': (1, 4)}\n",
        b"{'descr': '<f8', 'shape': (1, 4)}\n",
        b"{'descr': '<f8', 'fortran_order': False, 'shape': (True, 4)}\n",
        b"{'descr': '<f8', 'fortran_order': False, 'shape': (-1, 4)}\n",
        b"{'descr': '<f8', 'fortran_order': False, 'shape': (4,)}\n",
        b"{'descr': '<f8', 'fortran_order': 0, 'shape': (1, 4)}\n",
        b"{'descr': [('x', '<f8')], 'fortran_order': False, 'shape': (1, 4)}\n",
        b"{'descr': 'invalid', 'fortran_order': False, 'shape': (1, 4)}\n",
        b"{'descr': '<f8', 'fortran_order': False, 'shape': (1, 4)}",
        b"{invalid}\n",
    ],
)
def test_actual_npy_header_damage_refuses_without_payload(tmp_path: Path, header: bytes) -> None:
    """Ambiguous metadata and malformed scalar/shape/layout declarations refuse before allocation."""
    path = tmp_path / "events.npy"
    path.write_bytes(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header)
    with pytest.raises(ValueError):
        read_dvs_recording(path)


@pytest.mark.parametrize("damage", ["magic", "version", "header_size", "short_header"])
def test_incomplete_or_unsupported_npy_prefix_refuses(tmp_path: Path, damage: str) -> None:
    """Invalid preambles are not treated as archive or pickle formats."""
    path = tmp_path / "events.npy"
    raw = b"\x93NUMPY\x01\x00"
    if damage == "magic":
        raw = b"notNPY"
    elif damage == "version":
        raw = b"\x93NUMPY\x04\x00"
    elif damage == "header_size":
        raw += struct.pack("<H", 10001)
    else:
        raw += struct.pack("<H", 100) + b"short"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        read_dvs_recording(path)


def test_declared_huge_matrix_refuses_before_missing_payload_read(tmp_path: Path) -> None:
    """A huge rank-two declaration fails the matrix budget without allocating or requiring data."""
    path = tmp_path / "events.npy"
    header = (
        "{'descr': '<f8', 'fortran_order': False, 'shape': (" + str(1 << 60) + ", 4)}\n"
    ).encode()
    path.write_bytes(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header)
    with pytest.raises(ValueError, match="event budget"):
        read_dvs_recording(path)


@pytest.mark.parametrize("byteorder", ["<", ">"])
@pytest.mark.parametrize(
    "kind,width", [("i", 1), ("i", 2), ("i", 4), ("i", 8), ("u", 1), ("u", 2), ("u", 4), ("u", 8)]
)
@pytest.mark.parametrize("fortran", [False, True])
def test_integer_extremes_and_rounding_match_numpy(
    tmp_path: Path, byteorder: str, kind: str, width: int, fortran: bool
) -> None:
    """Native parity must retain signedness, byte order and float64 rounding at integer limits."""
    dtype = np.dtype(f"{byteorder}{kind}{width}")
    limits = np.iinfo(dtype)
    values = np.array(
        [[limits.min, limits.max, limits.max - 1, 0], [1, limits.min + 1, limits.max // 2, 2]],
        dtype=dtype,
        order="F" if fortran else "C",
    )
    path = tmp_path / "integers.npy"
    np.save(path, values, allow_pickle=False)
    actual = read_dvs_recording(path)
    expected = values.astype(np.float64)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize("byteorder", ["<", ">"])
@pytest.mark.parametrize("width", [2, 4, 8, 16])
@pytest.mark.parametrize("fortran", [False, True])
def test_float_extremes_signed_zero_and_nonfinite_values_match_numpy(
    tmp_path: Path, byteorder: str, width: int, fortran: bool
) -> None:
    """Decoding preserves floating conversion; finite event admission belongs to the caller."""
    dtype = np.dtype(f"{byteorder}f{width}")
    limits = np.finfo(dtype)
    values = np.array(
        [
            [0.0, -0.0, np.inf, -np.inf],
            [np.nan, limits.tiny, limits.smallest_subnormal, limits.max],
        ],
        dtype=dtype,
        order="F" if fortran else "C",
    )
    path = tmp_path / "floats.npy"
    np.save(path, values, allow_pickle=False)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        actual = read_dvs_recording(path)
        expected = values.astype(np.float64)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    finite_or_infinite = ~np.isnan(expected)
    np.testing.assert_array_equal(
        actual[finite_or_infinite].view(np.uint64), expected[finite_or_infinite].view(np.uint64)
    )


def test_extended_precision_rounding_uses_stored_values(tmp_path: Path) -> None:
    """Values around a float64 midpoint widen or round from the stored extended representation."""
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
    path = tmp_path / "extended.npy"
    np.save(path, values, allow_pickle=False)
    actual = read_dvs_recording(path)
    expected = values.astype(np.float64)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))

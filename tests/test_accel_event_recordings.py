# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native event-recording runtime parity

"""Exercise a compiled Go decoder through the real eager and manifest readers."""

from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from sc_neurocore.accel.event_recordings import decode_nmnist_recording
from sc_neurocore.datasets import load_nmnist
from sc_neurocore.datasets.encoders import EventBinning
from sc_neurocore.datasets.event_samples import read_event_sample
from sc_neurocore.datasets.manifest import build_manifest
from tests.event_dataset_support import nmnist_event_bytes, write_nmnist
from sc_neurocore.accel.mojo.isa_baseline import pin_isa


@pytest.fixture(scope="module")
def go_recording_library(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Compile the real Go C interface into a fixture-owned native library."""
    root = Path(__file__).resolve().parents[1]
    output = tmp_path_factory.mktemp("go-event-reader") / "libloaders.so"
    subprocess.run(
        ["go", "build", "-buildmode=c-shared", "-o", str(output), "./services/loaders/cshared"],
        cwd=root / "src/sc_neurocore/accel/go",
        check=True,
        capture_output=True,
        timeout=120,
    )
    return output


@pytest.fixture(scope="module")
def rust_recording_library(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the canonical Rust kernel with its own target directory."""
    root = Path(__file__).resolve().parents[1]
    target = tmp_path_factory.mktemp("rust-event-reader")
    manifest = root / "src/sc_neurocore/accel/rust/safety/nmnist_native/Cargo.toml"
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--locked",
            "--manifest-path",
            str(manifest),
            "--target-dir",
            str(target),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return target / "release/libsc_neurocore_nmnist.so"


@pytest.fixture(scope="module")
def mojo_recording_library(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the actual Mojo decoder with all compiler warnings treated as errors."""
    root = Path(__file__).resolve().parents[1]
    output = tmp_path_factory.mktemp("mojo-event-reader") / "libnmnist.so"
    subprocess.run(
        pin_isa(
            [
                "mojo",
                "build",
                "--Werror",
                "--fp-mode",
                "contract=off",
                "--emit",
                "shared-lib",
                "-o",
                str(output),
                str(root / "src/sc_neurocore/accel/mojo/kernels/nmnist.mojo"),
            ]
        ),
        check=True,
        capture_output=True,
        timeout=120,
    )
    return output


@pytest.fixture(scope="module", params=("go", "rust", "mojo"))
def native_recording_library(
    request: pytest.FixtureRequest,
    go_recording_library: Path,
    rust_recording_library: Path,
    mojo_recording_library: Path,
) -> tuple[Literal["go", "rust", "mojo"], Path]:
    """Select real compiled libraries for the same runtime boundary tests."""
    backend: Literal["go", "rust", "mojo"] = request.param
    return backend, {
        "go": go_recording_library,
        "rust": rust_recording_library,
        "mojo": mojo_recording_library,
    }[backend]


def test_native_recording_matches_numpy_bits(
    native_recording_library: tuple[Literal["go", "rust", "mojo"], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every byte field and fractional timestamp agrees with the NumPy decoder."""
    backend, library = native_recording_library
    monkeypatch.setenv(f"SC_NEUROCORE_DATASET_{backend.upper()}_LIBRARY", str(library))
    raw = np.random.default_rng(7).integers(0, 256, size=(4096, 5), dtype=np.uint8).tobytes()
    golden = decode_nmnist_recording(raw, backend="numpy")
    np.testing.assert_array_equal(decode_nmnist_recording(raw, backend=backend), golden)
    np.testing.assert_array_equal(decode_nmnist_recording(raw), golden)
    assert decode_nmnist_recording(b"", backend=backend).shape == (0, 4)


@pytest.mark.parametrize("lazy", [False, True])
def test_public_reader_uses_native_precision(
    tmp_path: Path,
    native_recording_library: tuple[Literal["go", "rust", "mojo"], Path],
    monkeypatch: pytest.MonkeyPatch,
    lazy: bool,
) -> None:
    """A native recording reaches the same temporal bin and window refusal."""
    backend, library = native_recording_library
    monkeypatch.setenv(f"SC_NEUROCORE_DATASET_{backend.upper()}_LIBRARY", str(library))
    write_nmnist(tmp_path, {"train": {0: 1}})
    sample = build_manifest("nmnist", tmp_path, version="generated-format-fixture").samples[0]
    path = tmp_path / sample.file
    path.write_bytes(nmnist_event_bytes([(0, 0, 0, 1002), (1, 0, 1, 2004)]))
    events = read_event_sample(tmp_path, "nmnist", sample) if lazy else load_nmnist(tmp_path)[0][0]
    bins, channels = EventBinning(1.002, 2, 34, 34, "merge").encode(events).nonzero()
    assert bins.tolist() == [1]
    assert channels.tolist() == [0]
    path.write_bytes(path.read_bytes() + b"\x00")
    with pytest.raises(ValueError, match="incomplete 40-bit"):
        if lazy:
            read_event_sample(tmp_path, "nmnist", sample)
        else:
            load_nmnist(tmp_path)


def test_explicit_native_backend_refuses_missing_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Configured native failure cannot silently become a NumPy run."""
    monkeypatch.setenv("SC_NEUROCORE_DATASET_GO_LIBRARY", str(tmp_path / "missing.so"))
    with pytest.raises(RuntimeError, match="existing absolute"):
        decode_nmnist_recording(bytes(5))
    np.testing.assert_array_equal(
        decode_nmnist_recording(bytes(5), backend="numpy"), [[0, 0, 0, 0]]
    )


def test_native_abi_refuses_incomplete_input_without_output_mutation(
    native_recording_library: tuple[Literal["go", "rust", "mojo"], Path],
) -> None:
    """A direct C caller receives refusal before destination writes."""
    library = ctypes.CDLL(str(native_recording_library[1]))
    decode = library.nmnist_decode_c
    decode.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
    ]
    decode.restype = ctypes.c_int
    raw = (ctypes.c_uint8 * 6)(0, 0, 0, 0, 0, 1)
    output = (ctypes.c_double * 4)(9, 9, 9, 9)
    assert decode(raw, 6, output, 4) == -1
    assert list(output) == [9, 9, 9, 9]
    assert decode(raw, 5, output, 3) == -1
    assert list(output) == [9, 9, 9, 9]
    assert decode(None, 5, output, 4) == -1
    assert decode(raw, 5, None, 4) == -1
    assert decode(raw, ctypes.c_size_t(-1).value, output, 4) == -1
    assert decode(raw, 5, output, ctypes.c_size_t(-1).value) == -1
    assert decode(ctypes.cast(output, ctypes.POINTER(ctypes.c_uint8)), 5, output, 4) == -1
    assert list(output) == [9, 9, 9, 9]
    assert decode(None, 0, None, 0) == 0
    unaligned = (ctypes.c_uint8 * 40)(*([9] * 40))
    offset = 1 if ctypes.addressof(unaligned) % ctypes.alignment(ctypes.c_double) == 0 else 0
    destination = ctypes.cast(ctypes.byref(unaligned, offset), ctypes.POINTER(ctypes.c_double))
    assert decode(raw, 5, destination, 4) == -1
    assert list(unaligned) == [9] * 40

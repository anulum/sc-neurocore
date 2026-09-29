# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia recording file parity

"""Compare actual Julia file decoding with the public NumPy recording decoder."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.accel.event_recordings import decode_nmnist_recording
from sc_neurocore.datasets import load_nmnist
from sc_neurocore.datasets.encoders import EventBinning
from sc_neurocore.datasets.event_samples import read_event_sample
from sc_neurocore.datasets.manifest import build_manifest
from tests.event_dataset_support import nmnist_event_bytes, write_nmnist


@pytest.mark.parametrize("channel", ["1.11", "release"])
def test_julia_file_decoder_matches_numpy(tmp_path: Path, channel: str) -> None:
    """Recorded byte fields and all fractional times survive native file I/O."""
    raw = np.random.default_rng(7).integers(0, 256, size=(4096, 5), dtype=np.uint8).tobytes()
    recording = tmp_path / "recording.bin"
    output = tmp_path / "decoded.f64"
    recording.write_bytes(raw)
    kernel = Path(__file__).resolve().parents[1] / "src/sc_neurocore/accel/julia/datasets/nmnist.jl"
    subprocess.run(
        [
            "julia",
            f"+{channel}",
            "--startup-file=no",
            "--check-bounds=yes",
            "--depwarn=error",
            "-e",
            "include(ARGS[1]); write(ARGS[3], NMNISTRecordings.read_nmnist(ARGS[2]))",
            str(kernel),
            str(recording),
            str(output),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    actual = np.fromfile(output, dtype=np.float64).reshape(-1, 4)
    np.testing.assert_array_equal(actual, decode_nmnist_recording(raw, backend="numpy"))


def test_julia_runtime_reads_public_event_recordings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Actual JuliaCall decoding reaches both public readers and temporal binning."""
    monkeypatch.setenv("SC_NEUROCORE_DATASET_JULIA_ENABLED", "1")
    raw = np.random.default_rng(7).integers(0, 256, size=(4096, 5), dtype=np.uint8).tobytes()
    golden = decode_nmnist_recording(raw, backend="numpy")
    np.testing.assert_array_equal(decode_nmnist_recording(raw, backend="julia"), golden)
    np.testing.assert_array_equal(decode_nmnist_recording(raw), golden)
    assert decode_nmnist_recording(b"", backend="julia").shape == (0, 4)
    write_nmnist(tmp_path, {"train": {0: 1}})
    sample = build_manifest("nmnist", tmp_path, version="generated-format-fixture").samples[0]
    path = tmp_path / sample.file
    path.write_bytes(nmnist_event_bytes([(0, 0, 0, 1002), (1, 0, 1, 2004)]))
    for events in (read_event_sample(tmp_path, "nmnist", sample), load_nmnist(tmp_path)[0][0]):
        bins, channels = EventBinning(1.002, 2, 34, 34, "merge").encode(events).nonzero()
        assert bins.tolist() == [1]
        assert channels.tolist() == [0]
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="incomplete 40-bit"):
        load_nmnist(tmp_path)
    monkeypatch.setenv("PYTHON_JULIACALL_PROJECT", str(tmp_path / "missing-project"))
    with pytest.raises(RuntimeError, match="absolute Julia project"):
        decode_nmnist_recording(raw, backend="julia")


def test_julia_opt_in_refuses_unconfigured_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit unavailable backend never substitutes NumPy or resolves packages."""
    monkeypatch.setenv("SC_NEUROCORE_DATASET_JULIA_ENABLED", "0")
    with pytest.raises(RuntimeError, match="explicit opt-in"):
        decode_nmnist_recording(bytes(5), backend="julia")
    monkeypatch.setenv("SC_NEUROCORE_DATASET_JULIA_ENABLED", "bad")
    with pytest.raises(RuntimeError, match="0 or 1"):
        decode_nmnist_recording(bytes(5), backend="julia")
    monkeypatch.setenv("SC_NEUROCORE_DATASET_JULIA_ENABLED", "1")
    monkeypatch.delenv("PYTHON_JULIACALL_EXE", raising=False)
    with pytest.raises(RuntimeError, match="absolute Julia executable"):
        decode_nmnist_recording(bytes(5), backend="julia")

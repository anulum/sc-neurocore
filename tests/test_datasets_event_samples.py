# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Individual event recording readers

"""Read actual camera and auditory files through the public sample reader."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.datasets.event_samples import read_event_sample
from sc_neurocore.datasets.manifest import SampleRecord, build_manifest
from tests.event_dataset_support import (
    nmnist_event_bytes,
    write_dvs_cifar10,
    write_nmnist,
    write_shd,
)


def test_nmnist_sensor_edges_polarity_and_microseconds(tmp_path: Path) -> None:
    """The public reader preserves all address and timestamp bits."""
    write_nmnist(tmp_path, {"train": {0: 1}})
    manifest = build_manifest("nmnist", tmp_path, version="generated-format-fixture")
    sample = manifest.samples[0]
    (tmp_path / sample.file).write_bytes(nmnist_event_bytes([(33, 32, 1, 1_234_567), (0, 0, 0, 0)]))
    np.testing.assert_array_equal(
        read_event_sample(tmp_path, "nmnist", sample),
        np.array([[33, 32, 1, 1234.567], [0, 0, 0, 0]], dtype=np.float64),
    )


def test_shd_selects_recording_and_converts_seconds(tmp_path: Path) -> None:
    """One selected HDF5 recording becomes channel events in milliseconds."""
    write_shd(tmp_path, {"train": [0, 1]})
    manifest = build_manifest("shd", tmp_path, version="generated-format-fixture")
    events = read_event_sample(tmp_path, "shd", manifest.samples[1])
    expected = np.array([[1, 0, 0, 1], [699, 0, 0, 2.5]], dtype=np.float64)
    np.testing.assert_allclose(events, expected, rtol=1e-6)


def test_dvs_cifar_preserves_declared_millisecond_events(tmp_path: Path) -> None:
    """The .npy sample is read without pickle or time rescaling."""
    write_dvs_cifar10(tmp_path, {"train": {2: 2}})
    manifest = build_manifest("dvs_cifar10", tmp_path, version="generated-format-fixture")
    np.testing.assert_array_equal(
        read_event_sample(tmp_path, "dvs_cifar10", manifest.samples[1]), [[1, 2, 1, 0.5]]
    )


@pytest.mark.parametrize("index", [-1, True, 1])
def test_camera_index_cannot_select_another_recording(tmp_path: Path, index: int) -> None:
    """Invalid indices fail instead of being ignored for a single-recording file."""
    write_nmnist(tmp_path, {"train": {0: 1}})
    manifest = build_manifest("nmnist", tmp_path, version="generated-format-fixture")
    with pytest.raises(ValueError, match="index|exactly one"):
        read_event_sample(tmp_path, "nmnist", replace(manifest.samples[0], index=index))


def test_path_outside_root_is_refused_before_read(tmp_path: Path) -> None:
    """A manifest sample cannot select another operator directory."""
    sample = SampleRecord("train", "../outside.bin", 0, 0, "recording")
    with pytest.raises(ValueError, match="outside"):
        read_event_sample(tmp_path, "nmnist", sample)


def test_incomplete_camera_record_is_refused(tmp_path: Path) -> None:
    """An incomplete 40-bit event cannot silently disappear."""
    (tmp_path / "recording.bin").write_bytes(b"\x00\x01")
    sample = SampleRecord("train", "recording.bin", 0, 0, "recording")
    with pytest.raises(ValueError, match="incomplete"):
        read_event_sample(tmp_path, "nmnist", sample)


def test_malformed_dvs_columns_are_refused(tmp_path: Path) -> None:
    """A camera array must carry all four event columns."""
    np.save(tmp_path / "recording.npy", np.zeros((2, 3)))
    sample = SampleRecord("train", "recording.npy", 0, 0, "recording")
    with pytest.raises(ValueError, match="four columns"):
        read_event_sample(tmp_path, "dvs_cifar10", sample)


def test_shd_index_outside_file_is_refused(tmp_path: Path) -> None:
    """An invalid recording index is rejected before HDF5 selection."""
    write_shd(tmp_path, {"train": [0]})
    manifest = build_manifest("shd", tmp_path, version="generated-format-fixture")
    with pytest.raises(ValueError, match="outside the recording"):
        read_event_sample(tmp_path, "shd", replace(manifest.samples[0], index=1))


def test_unknown_dataset_cannot_reuse_camera_reader(tmp_path: Path) -> None:
    """Dataset names have explicit supported semantics."""
    sample = SampleRecord("train", "recording", 0, 0, "recording")
    with pytest.raises(ValueError, match="unsupported"):
        read_event_sample(tmp_path, "unknown", sample)


def test_mismatched_auditory_event_vectors_are_refused(tmp_path: Path) -> None:
    """A corrupt HDF5 sample cannot pair unrelated channels and timestamps."""
    import h5py

    write_shd(tmp_path, {"train": [0]})
    manifest = build_manifest("shd", tmp_path, version="generated-format-fixture")
    with h5py.File(tmp_path / "shd_train.h5", "r+") as handle:
        handle["spikes/times"][0] = np.array([0.001], dtype=np.float32)
    with pytest.raises(ValueError, match="matching vectors"):
        read_event_sample(tmp_path, "shd", manifest.samples[0])


@pytest.mark.parametrize("lazy", [False, True])
def test_nmnist_fractional_bins_and_window_boundary(tmp_path: Path, lazy: bool) -> None:
    """Recorded microseconds retain their declared bin and out-of-window status."""
    from sc_neurocore.datasets import load_nmnist
    from sc_neurocore.datasets.encoders import EventBinning

    write_nmnist(tmp_path, {"train": {0: 1}})
    sample = build_manifest("nmnist", tmp_path, version="generated-format-fixture").samples[0]
    (tmp_path / sample.file).write_bytes(nmnist_event_bytes([(0, 0, 0, 1002), (1, 0, 0, 2004)]))
    if lazy:
        events = read_event_sample(tmp_path, "nmnist", sample)
    else:
        samples, labels = load_nmnist(tmp_path)
        assert labels.tolist() == [0]
        events = samples[0]
    spikes = EventBinning(1.002, 2, 34, 34, "merge").encode(events)
    positions, channels = spikes.nonzero()
    assert positions.tolist() == [1]
    assert channels.tolist() == [0]


def test_eager_nmnist_refuses_incomplete_record(tmp_path: Path) -> None:
    """The eager loader cannot discard a damaged final event either."""
    from sc_neurocore.datasets import load_nmnist

    recording = tmp_path / "Train" / "0" / "broken.bin"
    recording.parent.mkdir(parents=True)
    recording.write_bytes(nmnist_event_bytes([(0, 0, 0, 1002)]) + b"\x00")
    with pytest.raises(ValueError, match="incomplete 40-bit"):
        load_nmnist(tmp_path)


@pytest.mark.parametrize("lazy", [False, True])
def test_dvs_fractional_bins_and_window_boundary(tmp_path: Path, lazy: bool) -> None:
    """Both public readers preserve recorded times before encoding the window."""
    from sc_neurocore.datasets import load_dvs_cifar10
    from sc_neurocore.datasets.encoders import EventBinning

    write_dvs_cifar10(tmp_path, {"train": {0: 1}})
    sample = build_manifest("dvs_cifar10", tmp_path, version="generated-format-fixture").samples[0]
    recorded = np.array([[0, 0, 0, 1.002], [1, 0, 1, 2.004]], dtype=np.float64)
    np.save(tmp_path / sample.file, recorded)
    if lazy:
        events = read_event_sample(tmp_path, "dvs_cifar10", sample)
    else:
        samples, labels = load_dvs_cifar10(tmp_path)
        assert labels.tolist() == [0]
        events = samples[0]
    spikes = EventBinning(1.002, 2, 128, 128, "merge").encode(events)
    positions, channels = spikes.nonzero()
    assert positions.tolist() == [1]
    assert channels.tolist() == [0]
    np.testing.assert_array_equal(events, recorded)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("shape", [(2, 3), (4,), (1, 2, 4)])
def test_dvs_readers_refuse_incompatible_event_arrays(
    tmp_path: Path, lazy: bool, shape: tuple[int, ...]
) -> None:
    """Malformed recordings fail at either public reader before reaching training."""
    from sc_neurocore.datasets import load_dvs_cifar10

    write_dvs_cifar10(tmp_path, {"train": {0: 1}})
    sample = build_manifest("dvs_cifar10", tmp_path, version="generated-format-fixture").samples[0]
    np.save(tmp_path / sample.file, np.zeros(shape))
    with pytest.raises(ValueError, match="four columns"):
        if lazy:
            read_event_sample(tmp_path, "dvs_cifar10", sample)
        else:
            load_dvs_cifar10(tmp_path)


def test_shd_half_precision_recording_matches_training_bins(tmp_path: Path) -> None:
    """Stored float16 seconds are widened before either public path bins them."""
    import h5py

    from sc_neurocore.datasets import load_shd
    from sc_neurocore.datasets.encoders import EventBinning

    write_shd(tmp_path, {"train": [0]})
    with h5py.File(tmp_path / "shd_train.h5", "r+") as handle:
        del handle["spikes/times"]
        times = handle.create_dataset(
            "spikes/times", (1,), dtype=h5py.vlen_dtype(np.dtype("float16"))
        )
        times[0] = np.array([0.1, 0.25], dtype=np.float16)
    sample = build_manifest("shd", tmp_path, version="generated-format-fixture").samples[0]
    events = read_event_sample(tmp_path, "shd", sample)
    expected = EventBinning(1.0, 300, 700, 1, "merge").encode(events)
    eager, labels = load_shd(tmp_path, dt_ms=1.0, T=300)
    assert labels.tolist() == [0]
    np.testing.assert_array_equal(eager[0], expected[: len(eager[0])])

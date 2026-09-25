# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Small event-dataset trees in the published file formats

"""Write event-dataset directories in the layouts and formats the loaders read.

The files are real: N-MNIST events are packed into 40-bit records byte by
byte, SHD is an HDF5 file with the published groups, CIFAR10-DVS samples are
``.npy`` arrays. Only their size is small.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def nmnist_event_bytes(events: list[tuple[int, int, int, int]]) -> bytes:
    """Pack ``(x, y, polarity, microseconds)`` events as the published 40-bit records."""
    out = bytearray()
    for x, y, polarity, time_us in events:
        out += bytes(
            [
                x,
                y,
                (polarity << 7) | ((time_us >> 16) & 0x7F),
                (time_us >> 8) & 0xFF,
                time_us & 0xFF,
            ]
        )
    return bytes(out)


def write_nmnist(root: Path, per_class: dict[str, dict[int, int]]) -> None:
    """Write ``per_class[split][label]`` recordings per class, e.g. ``{"train": {0: 3}}``."""
    folders = {"train": "Train", "test": "Test"}
    for split, classes in per_class.items():
        for label, count in classes.items():
            directory = root / folders[split] / str(label)
            directory.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                events = [(index % 34, label, index % 2, 1000 * (index + 1))]
                (directory / f"{index:05d}.bin").write_bytes(nmnist_event_bytes(events))


def write_shd(root: Path, speakers: dict[str, list[int]]) -> None:
    """Write SHD files whose sample ``i`` of a split has speaker ``speakers[split][i]``."""
    import h5py

    root.mkdir(parents=True, exist_ok=True)
    files = {"train": "shd_train.h5", "test": "shd_test.h5"}
    for split, speaker_ids in speakers.items():
        with h5py.File(root / files[split], "w") as handle:
            float_list = h5py.vlen_dtype(np.dtype("float32"))
            int_list = h5py.vlen_dtype(np.dtype("uint16"))
            times = handle.create_dataset("spikes/times", (len(speaker_ids),), dtype=float_list)
            units = handle.create_dataset("spikes/units", (len(speaker_ids),), dtype=int_list)
            for index in range(len(speaker_ids)):
                times[index] = np.array([0.001, 0.0025], dtype=np.float32)
                units[index] = np.array([index % 700, 699], dtype=np.uint16)
            handle.create_dataset("labels", data=np.arange(len(speaker_ids)) % 20)
            handle.create_dataset("extra/speaker", data=np.asarray(speaker_ids, dtype=np.int64))


def write_dvs_cifar10(root: Path, per_class: dict[str, dict[int, int]]) -> None:
    """Write ``.npy`` event arrays in the layout the CIFAR10-DVS loader reads."""
    for split, classes in per_class.items():
        for label, count in classes.items():
            directory = root / split / str(label)
            directory.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                events = np.array([[index, label, 1, 0.5 * index]], dtype=np.float32)
                np.save(directory / f"{index:04d}.npy", events)

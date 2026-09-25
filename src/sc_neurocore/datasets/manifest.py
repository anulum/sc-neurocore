# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Versioned manifests of event-dataset files on disk

"""Record exactly which event-dataset files an experiment read.

A manifest names the dataset, its source and licence, the sensor it was
recorded with and the time unit its files store, then lists every file with
its size and SHA-256 and every sample with its published split, label and
group. The group is the unit that must not straddle a training and an
evaluation split — a speaker in SHD, a recording file where the publisher
names no finer identity — and :mod:`sc_neurocore.datasets.splits` divides
data by it.

Nothing here downloads data. The manifest describes files the user already
holds, and :func:`verify_manifest` says which of them are missing, changed or
not in the manifest, so a result can be tied to the bytes it was computed on.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA = "sc-neurocore.event-dataset.v1"

_CHUNK = 1 << 20


@dataclass(frozen=True, slots=True)
class DatasetDescription:
    """What one supported event dataset is, as its publisher states it.

    Attributes
    ----------
    name:
        Identifier used by the loaders and the manifest, e.g. ``"shd"``.
    title:
        Published name.
    citation:
        The paper to cite.
    doi:
        DOI of that paper.
    url:
        Where the publisher distributes the files.
    licence:
        SPDX identifier of the data licence.
    licence_url:
        Text of that licence.
    sensor:
        ``"dvs"`` for an event camera, ``"cochlea"`` for an auditory model.
    geometry:
        ``(width, height)`` of a camera, or ``(channels,)`` of a cochlea.
    polarities:
        Number of event polarities; ``0`` when events carry none.
    classes:
        Number of labels.
    file_format:
        The file format the loader reads, including the unit of its times.
    group_key:
        What a group is, and why it is the leakage unit.
    """

    name: str
    title: str
    citation: str
    doi: str
    url: str
    licence: str
    licence_url: str
    sensor: str
    geometry: tuple[int, ...]
    polarities: int
    classes: int
    file_format: str
    group_key: str

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON form."""
        return {
            "name": self.name,
            "title": self.title,
            "citation": self.citation,
            "doi": self.doi,
            "url": self.url,
            "licence": self.licence,
            "licence_url": self.licence_url,
            "sensor": self.sensor,
            "geometry": list(self.geometry),
            "polarities": self.polarities,
            "classes": self.classes,
            "file_format": self.file_format,
            "group_key": self.group_key,
        }


DATASETS: dict[str, DatasetDescription] = {
    "nmnist": DatasetDescription(
        name="nmnist",
        title="N-MNIST",
        citation=(
            "Orchard, G., Jayawant, A., Cohen, G. K. & Thakor, N. (2015). Converting static "
            "image datasets to spiking neuromorphic datasets using saccades. Front. Neurosci. 9:437"
        ),
        doi="10.3389/fnins.2015.00437",
        url="https://www.garrickorchard.com/datasets/n-mnist",
        licence="CC-BY-SA-4.0",
        licence_url="https://creativecommons.org/licenses/by-sa/4.0/",
        sensor="dvs",
        geometry=(34, 34),
        polarities=2,
        classes=10,
        file_format=(
            "one .bin file per sample under Train/<label>/ or Test/<label>/; 40-bit events: "
            "x byte, y byte, polarity bit, 23-bit timestamp in microseconds"
        ),
        group_key=(
            "recording file: each file is one MNIST image and the publisher names no writer, "
            "so no finer identity can be kept apart"
        ),
    ),
    "shd": DatasetDescription(
        name="shd",
        title="Spiking Heidelberg Digits",
        citation=(
            "Cramer, B., Stradmann, Y., Schemmel, J. & Zenke, F. (2022). The Heidelberg spiking "
            "data sets for the systematic evaluation of spiking neural networks. IEEE TNNLS "
            "33(7):2744-2757"
        ),
        doi="10.1109/TNNLS.2020.3044364",
        url="https://zenkelab.org/datasets/",
        licence="CC-BY-4.0",
        licence_url="https://creativecommons.org/licenses/by/4.0/",
        sensor="cochlea",
        geometry=(700,),
        polarities=0,
        classes=20,
        file_format=(
            "shd_train.h5 and shd_test.h5; spikes/times in seconds, spikes/units the channel, "
            "labels, extra/speaker"
        ),
        group_key="speaker: a model that heard a speaker in training is not tested on them",
    ),
    "dvs_cifar10": DatasetDescription(
        name="dvs_cifar10",
        title="CIFAR10-DVS",
        citation=(
            "Li, H., Liu, H., Ji, X., Li, G. & Shi, L. (2017). CIFAR10-DVS: an event-stream "
            "dataset for object classification. Front. Neurosci. 11:309"
        ),
        doi="10.3389/fnins.2017.00309",
        url="https://figshare.com/articles/dataset/CIFAR10-DVS/4724671",
        licence="CC-BY-4.0",
        licence_url="https://creativecommons.org/licenses/by/4.0/",
        sensor="dvs",
        geometry=(128, 128),
        polarities=2,
        classes=10,
        file_format=(
            "one .npy file per sample under train/<label>/ or test/<label>/, converted by the "
            "user from the published AEDAT recordings; columns x, y, polarity, timestamp in ms. "
            "The publisher defines no train/test split"
        ),
        group_key=(
            "recording file: each file is one CIFAR-10 image and the publisher names no finer "
            "identity"
        ),
    ),
}


@dataclass(frozen=True, slots=True)
class FileRecord:
    """One file of the dataset: path relative to the root, size and digest."""

    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class SampleRecord:
    """One sample: where it is, which published split it is in, its label and group."""

    split: str
    file: str
    index: int
    label: int
    group: str


@dataclass(frozen=True, slots=True)
class EventDatasetManifest:
    """Every file and sample of one event dataset as it lies on disk.

    Attributes
    ----------
    dataset:
        The dataset description.
    version:
        The release of the data the user holds, as its publisher names it.
    files:
        Every file read, sorted by path.
    samples:
        Every sample, in loader order.
    """

    dataset: DatasetDescription
    version: str
    files: tuple[FileRecord, ...]
    samples: tuple[SampleRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON form, with the schema identifier."""
        return {
            "schema": MANIFEST_SCHEMA,
            "dataset": self.dataset.to_dict(),
            "version": self.version,
            "files": [
                {"path": record.path, "bytes": record.bytes, "sha256": record.sha256}
                for record in self.files
            ],
            "samples": [
                {
                    "split": sample.split,
                    "file": sample.file,
                    "index": sample.index,
                    "label": sample.label,
                    "group": sample.group,
                }
                for sample in self.samples
            ],
        }

    @property
    def digest(self) -> str:
        """``sha256:`` over the canonical JSON form; identifies this exact manifest."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def splits(self) -> tuple[str, ...]:
        """Return the published split names, in first-seen order."""
        return tuple(dict.fromkeys(sample.split for sample in self.samples))


@dataclass(frozen=True, slots=True)
class ManifestVerification:
    """How the files on disk differ from a manifest.

    Attributes
    ----------
    missing:
        Files the manifest lists that are not on disk.
    changed:
        Files whose size or SHA-256 differs.
    unexpected:
        Files in the dataset layout that the manifest does not list.
    """

    missing: tuple[str, ...]
    changed: tuple[str, ...]
    unexpected: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """``True`` when the disk holds exactly the manifest's bytes."""
        return not (self.missing or self.changed or self.unexpected)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _description(name: str) -> DatasetDescription:
    try:
        return DATASETS[name]
    except KeyError:
        raise ValueError(
            f"unknown event dataset {name!r}; supported: {', '.join(sorted(DATASETS))}"
        ) from None


def _class_files(
    root: Path, split_dirs: Mapping[str, str], suffix: str
) -> Iterator[tuple[str, Path, int]]:
    for split, directory in split_dirs.items():
        split_root = root / directory
        if not split_root.is_dir():
            continue
        for class_dir in sorted(split_root.iterdir()):
            if not class_dir.is_dir() or not class_dir.name.isdigit():
                continue
            for path in sorted(class_dir.glob(f"*{suffix}")):
                if path.is_file():
                    yield split, path, int(class_dir.name)


_SHD_FILES = {"train": "shd_train.h5", "test": "shd_test.h5"}
_CLASS_LAYOUT = {
    "nmnist": ({"train": "Train", "test": "Test"}, ".bin"),
    "dvs_cifar10": ({"train": "train", "test": "test"}, ".npy"),
}


def _layout_files(name: str, root: Path) -> list[Path]:
    if name == "shd":
        return [root / file for file in _SHD_FILES.values() if (root / file).is_file()]
    split_dirs, suffix = _CLASS_LAYOUT[name]
    return [path for _, path, _ in _class_files(root, split_dirs, suffix)]


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _shd_samples(root: Path) -> list[SampleRecord]:
    import h5py

    samples: list[SampleRecord] = []
    for split, file in _SHD_FILES.items():
        path = root / file
        if not path.is_file():
            continue
        with h5py.File(path, "r") as handle:
            labels = handle["labels"][:]
            speakers = handle["extra"]["speaker"][:]
            if len(speakers) != len(labels):
                raise ValueError(f"{file}: {len(labels)} labels but {len(speakers)} speakers")
            samples.extend(
                SampleRecord(
                    split=split,
                    file=file,
                    index=index,
                    label=int(label),
                    group=f"speaker:{int(speaker)}",
                )
                for index, (label, speaker) in enumerate(zip(labels, speakers, strict=True))
            )
    return samples


def build_manifest(name: str, root: str | Path, *, version: str) -> EventDatasetManifest:
    """Scan a dataset directory and record its files and samples.

    Parameters
    ----------
    name:
        ``"nmnist"``, ``"shd"`` or ``"dvs_cifar10"``.
    root:
        Directory in the layout the matching loader reads.
    version:
        The release the files come from, as the publisher names it; it is
        recorded, not inferred, because the files do not carry it.

    Returns
    -------
    EventDatasetManifest
        The manifest.

    Raises
    ------
    ValueError
        On an unknown dataset, an empty version, a directory holding none of
        the dataset's files, or a label outside the dataset's classes.
    """
    description = _description(name)
    if not version.strip():
        raise ValueError("the dataset version must be stated")
    base = Path(root)
    paths = _layout_files(name, base)
    if not paths:
        raise ValueError(f"{base} holds no {description.title} files in the expected layout")
    files = tuple(
        FileRecord(path=_relative(base, path), bytes=path.stat().st_size, sha256=_sha256(path))
        for path in sorted(paths, key=lambda item: _relative(base, item))
    )
    if name == "shd":
        samples = _shd_samples(base)
    else:
        split_dirs, suffix = _CLASS_LAYOUT[name]
        samples = []
        for split, path, label in _class_files(base, split_dirs, suffix):
            relative = _relative(base, path)
            samples.append(
                SampleRecord(split=split, file=relative, index=0, label=label, group=relative)
            )
    bad = sorted({s.label for s in samples if not 0 <= s.label < description.classes})
    if bad:
        raise ValueError(f"labels {bad} are outside the {description.classes} classes")
    return EventDatasetManifest(
        dataset=description, version=version, files=files, samples=tuple(samples)
    )


def verify_manifest(manifest: EventDatasetManifest, root: str | Path) -> ManifestVerification:
    """Compare the files under ``root`` with a manifest, byte for byte.

    Parameters
    ----------
    manifest:
        The manifest an experiment recorded.
    root:
        Directory holding the dataset now.

    Returns
    -------
    ManifestVerification
        Missing, changed and unexpected files; ``ok`` when there are none.
    """
    base = Path(root)
    listed = {record.path: record for record in manifest.files}
    on_disk = {_relative(base, path) for path in _layout_files(manifest.dataset.name, base)}
    missing = sorted(path for path in listed if path not in on_disk)
    changed = sorted(
        path
        for path in listed
        if path in on_disk
        and (
            (base / path).stat().st_size != listed[path].bytes
            or _sha256(base / path) != listed[path].sha256
        )
    )
    unexpected = sorted(on_disk - set(listed))
    return ManifestVerification(
        missing=tuple(missing), changed=tuple(changed), unexpected=tuple(unexpected)
    )


def _require_keys(value: Any, keys: set[str], where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        found = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise ValueError(f"{where} must have exactly the fields {sorted(keys)}; found {found}")
    return value


def manifest_from_dict(data: Mapping[str, Any]) -> EventDatasetManifest:
    """Read a manifest's JSON form, refusing anything it does not define.

    Parameters
    ----------
    data:
        The parsed JSON.

    Returns
    -------
    EventDatasetManifest
        The manifest.

    Raises
    ------
    ValueError
        On another schema, a missing or unknown field, or a dataset
        description that differs from the one this version supports.
    """
    top = _require_keys(data, {"schema", "dataset", "version", "files", "samples"}, "manifest")
    if top["schema"] != MANIFEST_SCHEMA:
        raise ValueError(f"manifest schema {top['schema']!r} is not {MANIFEST_SCHEMA!r}")
    dataset_data = top["dataset"]
    name = dataset_data.get("name") if isinstance(dataset_data, Mapping) else None
    description = _description(str(name))
    if dataset_data != description.to_dict():
        raise ValueError(
            f"the manifest describes {name!r} differently from this version of SC-NeuroCore"
        )
    files = tuple(
        FileRecord(
            path=str(record["path"]), bytes=int(record["bytes"]), sha256=str(record["sha256"])
        )
        for record in (
            _require_keys(item, {"path", "bytes", "sha256"}, "file record") for item in top["files"]
        )
    )
    samples = tuple(
        SampleRecord(
            split=str(record["split"]),
            file=str(record["file"]),
            index=int(record["index"]),
            label=int(record["label"]),
            group=str(record["group"]),
        )
        for record in (
            _require_keys(item, {"split", "file", "index", "label", "group"}, "sample record")
            for item in top["samples"]
        )
    )
    return EventDatasetManifest(
        dataset=description, version=str(top["version"]), files=files, samples=samples
    )


__all__ = [
    "DATASETS",
    "MANIFEST_SCHEMA",
    "DatasetDescription",
    "EventDatasetManifest",
    "FileRecord",
    "ManifestVerification",
    "SampleRecord",
    "build_manifest",
    "manifest_from_dict",
    "verify_manifest",
]

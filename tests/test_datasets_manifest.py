# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Event-dataset manifests record and verify real files

"""A manifest names every file by its bytes and every sample by split, label and group."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from sc_neurocore.datasets import load_nmnist
from sc_neurocore.datasets.manifest import (
    DATASETS,
    MANIFEST_SCHEMA,
    build_manifest,
    manifest_from_dict,
    verify_manifest,
)
from tests.event_dataset_support import write_dvs_cifar10, write_nmnist, write_shd


def test_an_nmnist_manifest_lists_each_file_by_its_bytes(tmp_path: Path) -> None:
    write_nmnist(tmp_path, {"train": {0: 2, 7: 1}, "test": {3: 1}})
    manifest = build_manifest("nmnist", tmp_path, version="2015 release")

    paths = [record.path for record in manifest.files]
    assert paths == [
        "Test/3/00000.bin",
        "Train/0/00000.bin",
        "Train/0/00001.bin",
        "Train/7/00000.bin",
    ]
    record = manifest.files[1]
    content = (tmp_path / record.path).read_bytes()
    assert (record.bytes, record.sha256) == (len(content), hashlib.sha256(content).hexdigest())
    # Samples follow loader order, with the loader's labels.
    events, labels = load_nmnist(tmp_path, train=True)
    # The support files are real 40-bit records: the loader decodes them back.
    assert events[1].tolist() == [[1.0, 0.0, 1.0, 2.0]]
    train = [sample for sample in manifest.samples if sample.split == "train"]
    assert [sample.label for sample in train] == labels.tolist()
    assert train[0].group == train[0].file == "Train/0/00000.bin"
    assert manifest.splits() == ("train", "test")


def test_the_description_carries_the_publishers_terms() -> None:
    shd = DATASETS["shd"]
    assert (shd.licence, shd.geometry, shd.classes) == ("CC-BY-4.0", (700,), 20)
    assert "seconds" in shd.file_format
    nmnist = DATASETS["nmnist"]
    assert (nmnist.licence, nmnist.geometry, nmnist.polarities) == ("CC-BY-SA-4.0", (34, 34), 2)
    assert "microseconds" in nmnist.file_format
    assert "no train/test split" in DATASETS["dvs_cifar10"].file_format


def test_an_shd_manifest_groups_samples_by_speaker(tmp_path: Path) -> None:
    write_shd(tmp_path, {"train": [1, 1, 2, 3], "test": [3, 9]})
    manifest = build_manifest("shd", tmp_path, version="1.0")

    assert [record.path for record in manifest.files] == ["shd_test.h5", "shd_train.h5"]
    assert [(s.split, s.file, s.index, s.group) for s in manifest.samples] == [
        ("train", "shd_train.h5", 0, "speaker:1"),
        ("train", "shd_train.h5", 1, "speaker:1"),
        ("train", "shd_train.h5", 2, "speaker:2"),
        ("train", "shd_train.h5", 3, "speaker:3"),
        ("test", "shd_test.h5", 0, "speaker:3"),
        ("test", "shd_test.h5", 1, "speaker:9"),
    ]


def test_an_shd_file_whose_speakers_do_not_match_its_labels_is_refused(tmp_path: Path) -> None:
    import h5py

    write_shd(tmp_path, {"train": [1, 2]})
    with h5py.File(tmp_path / "shd_train.h5", "a") as handle:
        del handle["extra/speaker"]
        handle.create_dataset("extra/speaker", data=[1])
    with pytest.raises(ValueError, match="2 labels but 1 speakers"):
        build_manifest("shd", tmp_path, version="1.0")


def test_a_cifar10_dvs_manifest_reads_the_converted_arrays(tmp_path: Path) -> None:
    write_dvs_cifar10(tmp_path, {"train": {4: 2}})
    manifest = build_manifest("dvs_cifar10", tmp_path, version="figshare v2")
    assert [sample.label for sample in manifest.samples] == [4, 4]
    assert manifest.dataset.doi == "10.3389/fnins.2017.00309"


@pytest.mark.parametrize(
    ("name", "version", "message"),
    [
        ("mnist", "1", "unknown event dataset 'mnist'"),
        ("nmnist", "  ", "version must be stated"),
        ("nmnist", "1", "holds no N-MNIST files"),
    ],
)
def test_a_manifest_needs_a_known_dataset_a_version_and_files(
    tmp_path: Path, name: str, version: str, message: str
) -> None:
    (tmp_path / "Train" / "notes").mkdir(parents=True)
    (tmp_path / "Train" / "readme.txt").write_text("not a class")
    with pytest.raises(ValueError, match=message):
        build_manifest(name, tmp_path, version=version)


def test_a_label_outside_the_classes_is_refused(tmp_path: Path) -> None:
    write_nmnist(tmp_path, {"train": {12: 1}})
    with pytest.raises(ValueError, match=r"labels \[12\] are outside the 10 classes"):
        build_manifest("nmnist", tmp_path, version="1")


def test_verification_names_missing_changed_and_unlisted_files(tmp_path: Path) -> None:
    write_nmnist(tmp_path, {"train": {0: 3}})
    manifest = build_manifest("nmnist", tmp_path, version="1")
    assert verify_manifest(manifest, tmp_path).ok

    (tmp_path / "Train/0/00000.bin").unlink()
    changed = tmp_path / "Train/0/00001.bin"
    content = bytearray(changed.read_bytes())
    content[-1] ^= 0xFF  # same size, other bytes
    changed.write_bytes(bytes(content))
    write_nmnist(tmp_path, {"test": {1: 1}})

    report = verify_manifest(manifest, tmp_path)
    assert not report.ok
    assert report.missing == ("Train/0/00000.bin",)
    assert report.changed == ("Train/0/00001.bin",)
    assert report.unexpected == ("Test/1/00000.bin",)


def test_a_manifest_survives_its_json_form_and_keeps_its_digest(tmp_path: Path) -> None:
    write_shd(tmp_path, {"train": [1, 2]})
    manifest = build_manifest("shd", tmp_path, version="1.0")
    text = json.dumps(manifest.to_dict())
    again = manifest_from_dict(json.loads(text))
    assert again == manifest
    assert again.digest == manifest.digest
    assert manifest.digest.startswith("sha256:")


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda data: data.update(schema="other.v0"), "is not 'sc-neurocore.event-dataset.v1'"),
        (lambda data: data.update(extra=1), "manifest must have exactly the fields"),
        (lambda data: data["dataset"].update(licence="MIT"), "describes 'nmnist' differently"),
        (lambda data: data["files"][0].pop("sha256"), "file record must have exactly"),
        (lambda data: data["samples"][0].update(weight=1), "sample record must have exactly"),
        (lambda data: data["dataset"].update(name="cifar"), "unknown event dataset 'cifar'"),
    ],
)
def test_a_manifest_that_is_not_exactly_this_schema_is_refused(
    tmp_path: Path, edit: Callable[[dict[str, Any]], object], message: str
) -> None:
    write_nmnist(tmp_path, {"train": {0: 1}})
    data = build_manifest("nmnist", tmp_path, version="1").to_dict()
    assert data["schema"] == MANIFEST_SCHEMA
    edit(data)
    with pytest.raises(ValueError, match=message):
        manifest_from_dict(data)


def test_a_manifest_that_is_not_an_object_is_refused() -> None:
    not_an_object: Any = []
    with pytest.raises(ValueError, match="found list"):
        manifest_from_dict(not_an_object)


def test_a_directory_named_like_a_sample_is_not_a_sample(tmp_path: Path) -> None:
    write_nmnist(tmp_path, {"train": {0: 1}})
    (tmp_path / "Train/0/99999.bin").mkdir()
    manifest = build_manifest("nmnist", tmp_path, version="1")
    assert [record.path for record in manifest.files] == ["Train/0/00000.bin"]
    assert verify_manifest(manifest, tmp_path).ok

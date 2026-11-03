# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real backing-file and damaged native snapshot input contracts

"""Damage unmapped owned image copies and actual observation copies, preserving loaded libraries."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import cast

import pytest

from tools.go_coverage_image_inputs import (
    Image,
    executable_image_maps,
    original_abi_launch_arguments,
    read_images,
    validate_audited_image,
    validate_existing_observer,
    validate_invalid_snapshot_status,
    validate_snapshot_output,
)


def actual_phase_rows(job: Path, phase: str) -> list[dict[str, object]]:
    """Read the completed actual native observation stream for the requested original phase."""
    path = next((job / "process-counters" / phase).glob("*/observations.jsonl"))
    return [json.loads(line) for line in path.read_text().splitlines()]


def owned_unmapped_image(job: Path, output: Path) -> Image:
    """Copy a genuine Go backing file without loading it or declaring it mapped in this process."""
    original = next(
        image for image in read_images(job / "images.json", job).values() if image.backend == "go"
    )
    path = output / original.path.name
    shutil.copyfile(original.path, path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original.sha256
    return Image(path, original.sha256, original.backend, original.model)


def replace_unmapped_copy(image: Image) -> None:
    """Atomically replace only an owned unmapped copy with the same bytes and a new real inode."""
    replacement = image.path.with_suffix(".replacement")
    shutil.copyfile(image.path, replacement)
    replacement.replace(image.path)


@pytest.mark.parametrize("fault", ["digest", "metadata", "identity"])
def test_real_audited_file_observations_refuse_byte_metadata_and_identity_changes(
    native_job: Path, tmp_path: Path, fault: str
) -> None:
    """Capture actual descriptor metadata around explicit changes to an unmapped owned file."""
    image = owned_unmapped_image(native_job, tmp_path)
    previous = image.path.stat()
    loaded_identity = None
    if fault == "digest":
        data = bytearray(image.path.read_bytes())
        data[-1] ^= 255
        image.path.write_bytes(data)
        before = image.path.stat()
    elif fault == "metadata":
        before = previous
        os.utime(image.path, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1))
    else:
        loaded_identity = (previous.st_dev, previous.st_ino)
        replace_unmapped_copy(image)
        before = image.path.stat()
    digest = hashlib.sha256(image.path.read_bytes()).hexdigest()
    after = image.path.stat()
    with pytest.raises(RuntimeError, match="bytes or metadata changed|backing identity changed"):
        validate_audited_image(image, before, after, digest, loaded_identity)


@pytest.mark.parametrize("fault", ["identity", "bytes"])
def test_backing_recheck_refuses_changes_to_an_unmapped_owned_copy(
    native_job: Path, tmp_path: Path, fault: str
) -> None:
    """Neither a new real inode nor changed bytes can satisfy the original backing observation."""
    image = owned_unmapped_image(native_job, tmp_path)
    previous = image.path.stat()
    if fault == "identity":
        replace_unmapped_copy(image)
    else:
        data = bytearray(image.path.read_bytes())
        data[-1] ^= 255
        image.path.write_bytes(data)
    with pytest.raises(RuntimeError, match="Image backing identity changed|Image bytes changed"):
        image.verify_backing((previous.st_dev, previous.st_ino))


@pytest.mark.parametrize("fault", ["empty", "execute", "device", "inode"])
def test_damaged_actual_map_rows_cannot_claim_an_executable_image(
    native_job: Path, tmp_path: Path, fault: str
) -> None:
    """Alter copied real mapping rows, never the address space or loaded code of any process."""
    rows = actual_phase_rows(native_job, "before")
    load = next(row for row in rows if row["event"] == "pre_dlopen")
    snapshot = next(
        row for row in rows if row["event"] == "snapshot" and row["path"] == load["path"]
    )
    device, inode = cast(int, load["device"]), cast(int, load["inode"])
    maps = cast(list[str], snapshot["maps"]).copy()
    if fault == "empty":
        maps.clear()
    elif fault == "inode":
        owned = owned_unmapped_image(native_job, tmp_path)
        inode = owned.path.stat().st_ino
    else:
        damaged = []
        for line in maps:
            fields = line.split(maxsplit=5)
            if fault == "execute":
                fields[1] = fields[1].replace("x", "-")
            else:
                other = Path("/dev/null").stat().st_dev
                assert other != device
                fields[3] = f"{os.major(other):x}:{os.minor(other):x}"
            damaged.append(" ".join(fields))
        maps = damaged
    with pytest.raises(RuntimeError, match="expected executable mapping"):
        executable_image_maps((device, inode), "\n".join(maps))


def test_real_success_status_cannot_validate_an_invalid_snapshot_destination(
    native_job: Path,
) -> None:
    """A genuine successful snapshot status is refused when presented as invalid-path evidence."""
    snapshot = next(
        row for row in actual_phase_rows(native_job, "before") if row["event"] == "snapshot"
    )
    with pytest.raises(RuntimeError, match="Invalid native snapshot path was accepted"):
        validate_invalid_snapshot_status(cast(int, snapshot["status"]))


@pytest.mark.parametrize("fault", ["status", "metadata", "counters"])
def test_real_refusal_status_or_missing_copied_counter_files_cannot_pass(
    native_job: Path, tmp_path: Path, fault: str
) -> None:
    """Use an actual native refusal or remove bytes only from a copied successful snapshot."""
    snapshot = next(
        row for row in actual_phase_rows(native_job, "before") if row["event"] == "snapshot"
    )
    original = Path(cast(str, snapshot["destination"]))
    destination = tmp_path / "snapshot"
    shutil.copytree(original, destination)
    status = cast(int, snapshot["status"])
    if fault == "status":
        refused = next(
            row
            for row in actual_phase_rows(native_job, "uncovered-tag")
            if row["event"] == "snapshot"
        )
        status = cast(int, refused["status"])
        assert status != 0
    else:
        pattern = "covmeta.*" if fault == "metadata" else "covcounters.*"
        originals = list(original.glob(pattern))
        assert originals
        for path in destination.glob(pattern):
            path.unlink()
        assert all(path.is_file() for path in originals)
    with pytest.raises(RuntimeError, match="snapshot refused|omitted metadata or counters"):
        validate_snapshot_output(destination, status)


@pytest.mark.parametrize(
    "fault",
    [
        "short",
        "argv-type",
        "non-string",
        "short-argv",
        "prefix",
        "script",
        "phase",
        "count",
        "environment",
    ],
)
def test_damaged_actual_audit_arguments_cannot_admit_an_original_child_launch(
    native_job: Path, fault: str
) -> None:
    """Damage captured real Popen argument copies without inventing events or processes."""
    execution = json.loads((native_job / "execution.json").read_text())
    stream = native_job / "process-counters/tests" / str(execution["pid"]) / "observations.jsonl"
    original = stream.read_bytes()
    rows = [json.loads(line) for line in original.splitlines()]
    launch = next(row for row in rows if row["event"] == "abi_child_launch")
    arguments = deepcopy(cast(list[object], launch["audit_arguments"]))
    argv = cast(list[object], arguments[1])
    phase = "tests"
    if fault == "short":
        arguments = arguments[:1]
    elif fault == "argv-type":
        arguments[1] = tuple(argv)
    elif fault == "non-string":
        argv[-1] = execution["pid"]
    elif fault == "short-argv":
        arguments[1] = argv[:3]
    elif fault == "prefix":
        argv[1] = argv[2]
    elif fault == "script":
        argv[3] = cast(str, argv[3]).replace("from tests.test_mat_family_c_abi import", "", 1)
    elif fault == "phase":
        phase = json.loads((native_job / "environments/before.json").read_text())[
            "NEUROCORE_GO_COVERAGE_PHASE"
        ]
    elif fault == "count":
        arguments.append(arguments[0])
    else:
        arguments[3] = json.loads((native_job / "environments/tests.json").read_text())
    if fault in {"phase", "count", "environment"}:
        with pytest.raises(RuntimeError, match="must inherit the declared test environment"):
            original_abi_launch_arguments(tuple(arguments), phase)
    else:
        assert original_abi_launch_arguments(tuple(arguments), phase) is None
    assert stream.read_bytes() == original


@pytest.mark.parametrize("fault", ["job", "phase"])
def test_actual_observer_scope_cannot_be_relabelled_by_a_different_request(
    native_job: Path, tmp_path: Path, fault: str
) -> None:
    """Compare real job/phase declarations without creating or mutating a process observer."""
    actual = json.loads((native_job / "environments/before.json").read_text())
    other = json.loads((native_job / "environments/contracts.json").read_text())
    actual_job = Path(actual["NEUROCORE_GO_COVERAGE_JOB"])
    actual_phase = actual["NEUROCORE_GO_COVERAGE_PHASE"]
    requested_job = tmp_path if fault == "job" else actual_job
    requested_phase = other["NEUROCORE_GO_COVERAGE_PHASE"] if fault == "phase" else actual_phase
    with pytest.raises(RuntimeError, match="another gate phase is already active"):
        validate_existing_observer(actual_job, actual_phase, requested_job, requested_phase)

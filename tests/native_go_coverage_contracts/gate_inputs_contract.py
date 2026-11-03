# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native producer custody and damaged observation contracts

"""Refuse changed owned file copies and damaged actual compiler, profile and source observations."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
from typing import cast

import pytest

from tools.go_coverage_gate_inputs import (
    BaselineSource,
    validate_before_work_sources,
    validate_bootstrap_custody,
    validate_compiler_version,
    validate_final_images,
    validate_image_transfer,
    validate_source_custody,
)
from tools.go_coverage_images import read_images
from tools.go_coverage_profiles import TARGET_STATEMENTS
from tools.go_mat_coverage_build import digest


@pytest.mark.parametrize("fault", ["producer", "copy"])
def test_actual_transfer_guard_refuses_changes_to_unmapped_owned_file_copies(
    native_job: Path, tmp_path: Path, fault: str
) -> None:
    """Measure actual source and destination changes after copying an unmapped real image."""
    image = next(
        i
        for i in read_images(native_job / "images.json", native_job).values()
        if i.backend == "mojo"
    )
    original, copied = tmp_path / "source.so", tmp_path / "copied.so"
    shutil.copyfile(image.path, original)
    before = digest(original)
    shutil.copyfile(original, copied)
    damaged = original if fault == "producer" else copied
    data = bytearray(damaged.read_bytes())
    data[-1] ^= 255
    damaged.write_bytes(data)
    with pytest.raises(RuntimeError, match="changed during input transfer"):
        validate_image_transfer(original, copied, before)
    assert digest(image.path) == image.sha256


@pytest.mark.parametrize("fault", ["missing", "different"])
def test_actual_compiler_stdout_cannot_match_a_damaged_module_declaration(
    native_job: Path, native_repository: Path, fault: str
) -> None:
    """Keep real compiler output intact while only a copied module version declaration changes."""
    module = (native_repository / "src/sc_neurocore/accel/go/go.mod").read_text()
    version = re.search(r"^go (\d+)\.(\d+)\.(\d+)$", module, re.MULTILINE)
    assert version is not None
    replacement = (
        "" if fault == "missing" else f"go {version[1]}.{version[2]}.{int(version[3]) + 1}"
    )
    module = module[: version.start()] + replacement + module[version.end() :]
    observed = (native_job / "commands/go-version/stdout.log").read_text()
    with pytest.raises(RuntimeError, match="maintained module pin"):
        validate_compiler_version(module, observed)


@pytest.mark.parametrize("fault", ["missing", "covered", "absent", "shrunk"])
def test_damaged_before_work_observations_cannot_claim_zero_complete_subjects(
    native_job: Path, fault: str
) -> None:
    """Use actual completed counters or damage only a copy of the actual before-work profile."""
    verdict = json.loads((native_job / "artifacts/before-work-verdict.json").read_text())
    rows = cast(list[BaselineSource], verdict["observed_after"]["sources"])
    if fault == "missing":
        rows.pop()
    elif fault == "covered":
        complete = json.loads(
            (native_job / "artifacts/complete-statement-verdict.json").read_text()
        )
        rows = cast(list[BaselineSource], complete["observed_after"]["sources"])
    elif fault == "absent":
        rows[0]["present"] = False
    else:
        rows[0]["total_statements"] = TARGET_STATEMENTS[rows[0]["path"]] - 1
    with pytest.raises(RuntimeError, match="missing, shrunk or already covered"):
        validate_before_work_sources(rows)


def test_final_image_guard_refuses_changed_unmapped_copy_and_preserves_originals(
    native_job: Path, tmp_path: Path
) -> None:
    """The unchanged real manifest is copied before a single owned backing file is damaged."""
    images = json.loads((native_job / "images.json").read_text())
    original = Path(images[0]["path"])
    before = digest(original)
    copy = tmp_path / original.name
    shutil.copyfile(original, copy)
    images[0]["path"] = str(copy)
    data = bytearray(copy.read_bytes())
    data[-1] ^= 255
    copy.write_bytes(data)
    with pytest.raises(RuntimeError, match="image changed during qualification"):
        validate_final_images(images)
    assert digest(original) == before


def test_final_bootstrap_guard_refuses_changed_serialized_observation(native_job: Path) -> None:
    """A damaged copy cannot replace the exact original bootstrap declaration bytes."""
    before = (native_job / "bootstrap-before.json").read_bytes()
    with pytest.raises(RuntimeError, match="child bootstrap changed"):
        validate_bootstrap_custody(before, before + b"\n")


def test_final_source_guard_refuses_omission_from_actual_source_pins(native_job: Path) -> None:
    """Every genuine original source pin remains required in the separately copied final mapping."""
    before = json.loads((native_job / "source-before.json").read_text())
    after = before.copy()
    after.pop(next(iter(after)))
    with pytest.raises(RuntimeError, match="source or producer executable changed"):
        validate_source_custody(before, after)

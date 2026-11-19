# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native proof storage metadata bindings

"""Bind retained metadata to its store and claimed resources to the actual native proof."""

from __future__ import annotations

import hashlib
import json
from contextlib import ExitStack
from pathlib import Path

import pytest

from tests.test_tools.test_function_synthesis_retention import RetentionCase, _copy_case
from tests.test_tools.test_function_synthesis_retention import retention_case as retention_case
from tools.function_synthesis_netlist import object_mapping
from tools.function_synthesis_records import SynthesisCase
from tools.function_synthesis_retention import materialize_artifacts


def _snapshot(root: Path) -> dict[str, tuple[str, int]]:
    """Retain actual native store identities without restoring or modifying their contents."""
    return {
        path.relative_to(root).as_posix(): (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_mtime_ns,
        )
        for path in root.rglob("*")
        if path.is_file()
    }


def test_materialization_refuses_changed_metadata_without_restamping(
    retention_case: RetentionCase, tmp_path: Path
) -> None:
    """Refuse metadata whose stored digest differs before a consumer can trust its native view."""
    _, original, _ = retention_case
    original_before = _snapshot(original.directory)
    _, entry = _copy_case(retention_case, tmp_path / "changed-metadata")
    path = entry / "measurement.json"
    path.write_bytes(path.read_bytes() + b"\n")
    before = _snapshot(entry)
    with pytest.raises(ValueError, match="Retained proof metadata changed"), ExitStack() as views:
        views.enter_context(materialize_artifacts(entry))
    assert _snapshot(entry) == before
    assert _snapshot(original.directory) == original_before


def test_coherent_storage_metadata_cannot_replace_actual_native_resources(
    retention_case: RetentionCase, tmp_path: Path
) -> None:
    """Recompute physical counts instead of accepting a forged claim with valid storage hashes."""
    _, original, _ = retention_case
    original_before = _snapshot(original.directory)
    cache, entry = _copy_case(retention_case, tmp_path / "forged-resource-count")
    path = entry / "measurement.json"
    document = object_mapping(json.loads(path.read_text()))
    measurement = object_mapping(document["measurement"])
    resources = object_mapping(measurement["resources"])
    count = resources["total_cells"]
    assert type(count) is int
    resources["total_cells"] = count + 1
    measurement["resources"] = resources
    document["measurement"] = measurement
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    manifest_path = entry / "retained-artifacts.json"
    manifest = object_mapping(json.loads(manifest_path.read_text()))
    metadata = object_mapping(manifest["metadata_sha256"])
    metadata["measurement.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["metadata_sha256"] = metadata
    manifest_path.write_text(json.dumps(manifest))
    before = _snapshot(entry)
    with pytest.raises(ValueError, match="Successful cache artifacts differ"):
        cache.get(SynthesisCase("floor", 8, 0))
    assert _snapshot(entry) == before
    assert not (cache.directory / "attempts").exists()
    assert _snapshot(original.directory) == original_before

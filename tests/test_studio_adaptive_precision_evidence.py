# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio adaptive precision evidence tests

"""Tests for Studio adaptive precision evidence classification contracts."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")

from starlette.testclient import TestClient

from sc_neurocore.compiler.adaptive_precision import (
    assign_synapse_precisions,
    write_precision_formal_evidence_bundle,
)
from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.evidence_classification import (
    validate_studio_evidence_classification,
    validate_studio_evidence_status,
)
from sc_neurocore.studio.platform.evidence_bundle import JsonValue


def test_adaptive_precision_formal_bundle_uses_studio_compile_evidence(
    tmp_path: Path,
) -> None:
    """Compiler formal bundles expose controlled Studio evidence fields."""

    assignments = assign_synapse_precisions(
        [np.array([[0.25, 0.75]])],
        target_error=0.01,
        min_bits=2,
        max_bits=8,
        min_length=16,
        max_length=256,
    )

    manifest = write_precision_formal_evidence_bundle(
        tmp_path,
        assignments,
        module_name="adaptive_precision_plan",
    )

    assert (
        validate_studio_evidence_classification(cast(str, manifest["evidence_classification"]))
        == "compile"
    )
    assert validate_studio_evidence_status(cast(str, manifest["status"])) == "completed"


def test_adaptive_precision_formal_bundle_rejects_empty_evidence(tmp_path: Path) -> None:
    """Compiler formal bundles fail closed when no precision evidence exists."""

    with pytest.raises(ValueError, match="assignments must not be empty"):
        write_precision_formal_evidence_bundle(tmp_path, [])


def test_studio_adaptive_precision_formal_bundle_route_returns_controlled_evidence() -> None:
    """Studio formal-bundle route returns evidence metadata on the public payload."""

    client = TestClient(create_app(), base_url="http://127.0.0.1")
    response = client.post(
        "/api/adaptive-precision/formal-bundle",
        json={
            "layer_weights": [[[0.2, 0.3], [0.4, 0.6]]],
            "layer_names": ["dense0"],
            "target_error_percent": 0.1,
            "module_name": "precision_plan_demo",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    bundle_manifest = cast(dict[str, JsonValue], payload["bundle_manifest"])

    assert (
        validate_studio_evidence_classification(
            cast(str, bundle_manifest["evidence_classification"])
        )
        == "compile"
    )
    assert validate_studio_evidence_status(cast(str, bundle_manifest["status"])) == "completed"
    assert bundle_manifest["evidence_boundary"] == (
        "bundle_generation_only_no_symbiyosys_execution_no_silicon_claim"
    )

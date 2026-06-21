# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio project save manifest tests

"""Tests for path-free Studio project save manifests."""

from __future__ import annotations

import hashlib
import json
import math

import pytest

from sc_neurocore.studio.project_manifest import (
    STUDIO_PROJECT_SAVE_SCHEMA_VERSION,
    StudioProjectSaveManifest,
    build_project_save_manifest,
    dump_project_payload,
)


def test_build_project_save_manifest_returns_path_free_digests() -> None:
    """Project save manifests expose stable evidence metadata without paths."""

    state = {"duration": 25.0, "graph": {"populations": []}}
    payload = {
        "name": "demo",
        "saved_at": 1_750_374_000.0,
        "version": "0.3.0",
        "state": state,
    }

    manifest = build_project_save_manifest(
        name="demo",
        saved_at=1_750_374_000.0,
        version="0.3.0",
        state=state,
        project_payload=payload,
    )
    public = manifest.to_public_dict()

    assert public == {
        "evidence_classification": "project_workspace",
        "name": "demo",
        "project_sha256": hashlib.sha256(
            json.dumps(
                payload,
                allow_nan=False,
                default=str,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
        "saved_at": 1_750_374_000.0,
        "schema_version": STUDIO_PROJECT_SAVE_SCHEMA_VERSION,
        "state_sha256": hashlib.sha256(
            json.dumps(
                state,
                allow_nan=False,
                default=str,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
        "version": "0.3.0",
    }
    assert "path" not in public


def test_dump_project_payload_keeps_readable_portable_json() -> None:
    """Durable project JSON is readable and rejects non-portable numbers."""

    payload = {"name": "demo", "saved_at": 1.0, "version": "0.3.0", "state": {}}

    encoded = dump_project_payload(payload)

    assert encoded.startswith("{\n")
    assert json.loads(encoded) == payload


def test_project_save_manifest_rejects_unknown_evidence_classification() -> None:
    """Project save manifests use the shared Studio evidence-class contract."""

    manifest = StudioProjectSaveManifest(
        name="demo",
        saved_at=1.0,
        version="0.3.0",
        state_sha256="0" * 64,
        project_sha256="1" * 64,
        evidence_classification="screenshots",  # type: ignore[arg-type]  # Invalid by design.
    )

    with pytest.raises(ValueError, match="classification"):
        manifest.to_public_dict()


@pytest.mark.parametrize(
    "payload",
    [
        {"bad": math.nan},
        {"bad": math.inf},
    ],
)
def test_project_manifest_rejects_non_portable_json_numbers(
    payload: dict[str, float],
) -> None:
    """Project manifests fail closed on JSON values that are not portable."""

    with pytest.raises(ValueError, match="portable JSON"):
        dump_project_payload(payload)
    with pytest.raises(ValueError, match="portable JSON"):
        build_project_save_manifest(
            name="demo",
            saved_at=1.0,
            version="0.3.0",
            state=payload,
            project_payload=payload,
        )

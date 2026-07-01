# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — studio schema-A manifest artifact drift guard

"""Drift guard + envelope shape checks for the emitted studio CapabilityManifest.

The committed ``docs/_generated/studio_manifest.json`` is the federation-gate envelope
the SCPN-STUDIO keeper reviews. These tests keep it in lock-step with
:func:`sc_neurocore.federation.manifest.build_manifest` plus the real architecture-map
generator and assert the envelope shape the Hub split gate requires. Guarded by
``pytest.importorskip`` so they skip cleanly without the optional ``federation`` extra.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("scpn_studio_platform")

from sc_neurocore import __version__ as SOURCE_VERSION  # noqa: E402
from sc_neurocore.federation.manifest import STUDIO_VERSION  # noqa: E402

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_emitter() -> Any:
    path = _repo_root() / "tools" / "emit_studio_manifest.py"
    spec = importlib.util.spec_from_file_location("emit_studio_manifest", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_committed_artifact_matches_the_producer() -> None:
    emitter = _load_emitter()
    assert emitter._ARTIFACT.exists(), "run `python tools/emit_studio_manifest.py`"
    committed = json.loads(emitter._ARTIFACT.read_text(encoding="utf-8"))
    produced = json.loads(emitter.render())
    assert committed == produced, (
        "docs/_generated/studio_manifest.json is stale; run `python tools/emit_studio_manifest.py`"
    )


def test_manifest_default_version_matches_source_package() -> None:
    assert STUDIO_VERSION == SOURCE_VERSION


def test_artifact_is_schema_a_envelope_well_formed() -> None:
    envelope = json.loads(
        (_repo_root() / "docs" / "_generated" / "studio_manifest.json").read_text()
    )
    assert set(envelope) == {"architecture_map", "schema_a"}
    payload = envelope["schema_a"]
    assert payload["studio"] == "sc-neurocore"
    assert payload["studio_version"] == SOURCE_VERSION
    assert payload["contract_era"].startswith("v")
    assert payload["platform_sdk"] == ">=0.9,<0.10"
    assert _DIGEST_RE.match(payload["content_digest"]), payload["content_digest"]
    verbs = [verb["verb"] if isinstance(verb, dict) else verb for verb in payload["verbs"]]
    assert len(verbs) == len(set(verbs)) == 8, "the SC-NeuroCore vertical advertises eight verbs"
    evidence_types = payload["evidence_types"]
    assert all(schema.endswith(".v1") for schema in evidence_types)
    assert len(evidence_types) == len(set(evidence_types)) == 7

    architecture_map = envelope["architecture_map"]
    assert architecture_map["version"] == "architecture-map.v2"
    assert {"backends", "capabilities", "interfaces", "cross_repo", "boundaries"} <= set(
        architecture_map
    )
    assert any(backend["name"] == "rust" for backend in architecture_map["backends"])
    assert any(interface["kind"] == "rest" for interface in architecture_map["interfaces"])
    assert any(edge["sibling"] == "SCPN-CONTROL" for edge in architecture_map["cross_repo"])

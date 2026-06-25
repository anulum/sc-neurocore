# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — studio schema-A manifest artifact drift guard

"""Drift guard + schema-A shape checks for the emitted studio CapabilityManifest.

The committed ``docs/_generated/studio_manifest.json`` is the federation-gate artifact
the SCPN-STUDIO keeper reviews. These tests keep it in lock-step with
:func:`sc_neurocore.federation.manifest.build_manifest` (so a verb or evidence-schema
change cannot leave a stale federation manifest) and assert the schema-A shape the Hub
gate requires. Guarded by ``pytest.importorskip`` so they skip cleanly without the
optional ``federation`` extra.
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
    # studio_version is an environment-dependent stamp (installed distribution version
    # vs "0+unknown" from a source tree as in CI), so it is excluded — the structural
    # contract (verbs, evidence, digest, era) stays in lock-step, and content_digest is
    # computed over verbs+evidence, not studio_version.
    emitter = _load_emitter()
    assert emitter._ARTIFACT.exists(), "run `python tools/emit_studio_manifest.py`"
    committed = json.loads(emitter._ARTIFACT.read_text(encoding="utf-8"))
    produced = json.loads(emitter.render())
    committed.pop("studio_version", None)
    produced.pop("studio_version", None)
    assert committed == produced, (
        "docs/_generated/studio_manifest.json is stale; run `python tools/emit_studio_manifest.py`"
    )


def test_artifact_is_schema_a_well_formed() -> None:
    payload = json.loads(
        (_repo_root() / "docs" / "_generated" / "studio_manifest.json").read_text()
    )
    assert payload["studio"] == "sc-neurocore"
    assert payload["contract_era"].startswith("v")
    assert payload["platform_sdk"] == ">=0.9,<0.10"
    assert _DIGEST_RE.match(payload["content_digest"]), payload["content_digest"]
    verbs = [verb["verb"] if isinstance(verb, dict) else verb for verb in payload["verbs"]]
    assert len(verbs) == len(set(verbs)) == 8, "the SC-NeuroCore vertical advertises eight verbs"
    evidence_types = payload["evidence_types"]
    assert all(schema.endswith(".v1") for schema in evidence_types)
    assert len(evidence_types) == len(set(evidence_types)) == 7

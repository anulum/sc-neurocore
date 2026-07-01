#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — studio schema-A CapabilityManifest emitter

"""Emit (or check) the SC-NeuroCore studio federation envelope artifact.

This is the federation-gate artifact the SCPN-STUDIO keeper consumes — the schema-A
manifest carrying ``contract_era`` + ``evidence_types`` + ``verbs`` + ``content_digest``,
wrapped with the repository's real ``architecture-map.v2`` block for hub federation.

Requires the optional ``federation`` extra (``scpn-studio-platform``). ``--check`` fails
if the committed artifact has drifted from the producer, so a verb or evidence-schema
change cannot silently leave a stale federation manifest behind.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Protocol, cast

from sc_neurocore.federation.manifest import build_manifest

_ARTIFACT = Path(__file__).resolve().parents[1] / "docs" / "_generated" / "studio_manifest.json"


class _CapabilityManifestModule(Protocol):
    """Typed subset loaded from ``tools/capability_manifest.py``."""

    def build_architecture_map(self, repo: Path) -> dict[str, Any]:
        """Build the checked architecture-map block for the repository."""


def _load_capability_manifest_module() -> _CapabilityManifestModule:
    """Load the repository capability-manifest helper without requiring ``tools`` as a package."""

    path = Path(__file__).resolve().with_name("capability_manifest.py")
    spec = importlib.util.spec_from_file_location("_sc_neurocore_capability_manifest", path)
    if spec is None or spec.loader is None:  # pragma: no cover - importlib contract guard.
        raise RuntimeError(f"cannot load capability manifest helper from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return cast(_CapabilityManifestModule, module)


def build_architecture_map_extension() -> dict[str, Any]:
    """Return the real architecture-map.v2 block used by the Studio federation envelope."""

    repo = Path(__file__).resolve().parents[1]
    return _load_capability_manifest_module().build_architecture_map(repo)


def _normalise_env_dependent_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove environment-stamped fields that do not affect the Studio contract digest."""

    schema_a = payload.get("schema_a")
    if isinstance(schema_a, dict):
        schema_a.pop("studio_version", None)
    return payload


def render() -> str:
    """Return the deterministic Studio federation envelope JSON.

    Returns
    -------
    str
        The schema-A :class:`~scpn_studio_platform.manifest.CapabilityManifest`
        and architecture-map block serialised as sorted-key JSON with a trailing
        newline.
    """
    payload = {
        "schema_a": build_manifest().to_dict(),
        "architecture_map": build_architecture_map_extension(),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Emit the artifact, or check the committed copy against the producer.

    Parameters
    ----------
    argv
        Command-line arguments; defaults to ``sys.argv``.

    Returns
    -------
    int
        ``0`` on success, ``1`` when ``--check`` finds a missing or stale artifact.
    """
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the committed artifact differs from the producer (no write).",
    )
    args = parser.parse_args(argv)

    rendered = render()
    if args.check:
        if not _ARTIFACT.exists():
            print(f"{_ARTIFACT} is missing; run `python tools/emit_studio_manifest.py`.")
            return 1
        # ``schema_a.studio_version`` is an environment-dependent stamp (the installed
        # distribution version vs "0+unknown" from a source tree), excluded so the
        # check is env-stable; content_digest covers the verbs+evidence contract.
        committed = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
        produced = json.loads(rendered)
        if _normalise_env_dependent_fields(committed) != _normalise_env_dependent_fields(produced):
            print(f"{_ARTIFACT} is stale; run `python tools/emit_studio_manifest.py`.")
            return 1
        return 0

    _ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    _ARTIFACT.write_text(rendered, encoding="utf-8")
    print(f"wrote {_ARTIFACT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — studio schema-A CapabilityManifest emitter

"""Emit (or check) the SC-NeuroCore schema-A studio CapabilityManifest artifact.

This is the federation-gate artifact the SCPN-STUDIO keeper consumes — the schema-A
manifest carrying ``contract_era`` + ``evidence_types`` + ``verbs`` + ``content_digest``.
It is distinct from ``docs/_generated/capability_manifest.json`` (the repo-inventory
manifest with its ``architecture_map`` block); this one is the canonical product of
:func:`sc_neurocore.federation.manifest.build_manifest`.

Requires the optional ``federation`` extra (``scpn-studio-platform``). ``--check`` fails
if the committed artifact has drifted from the producer, so a verb or evidence-schema
change cannot silently leave a stale federation manifest behind.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sc_neurocore.federation.manifest import build_manifest

_ARTIFACT = Path(__file__).resolve().parents[1] / "docs" / "_generated" / "studio_manifest.json"


def render() -> str:
    """Return the deterministic schema-A manifest JSON (sorted, trailing newline).

    Returns
    -------
    str
        The schema-A :class:`~scpn_studio_platform.manifest.CapabilityManifest`
        serialised as sorted-key JSON with a trailing newline.
    """
    payload = build_manifest().to_dict()
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
        # ``studio_version`` is an environment-dependent stamp (the installed
        # distribution version vs "0+unknown" from a source tree), excluded so the
        # check is env-stable; content_digest covers the verbs+evidence contract.
        committed = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
        produced = json.loads(rendered)
        committed.pop("studio_version", None)
        produced.pop("studio_version", None)
        if committed != produced:
            print(f"{_ARTIFACT} is stale; run `python tools/emit_studio_manifest.py`.")
            return 1
        return 0

    _ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    _ARTIFACT.write_text(rendered, encoding="utf-8")
    print(f"wrote {_ARTIFACT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

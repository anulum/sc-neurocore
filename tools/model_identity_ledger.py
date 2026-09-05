#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Generated per-model identity and gap ledger

"""Emit or check the generated model identity ledger.

The ledger (``docs/_generated/model_identity_ledger.json``) is a projection of
:mod:`sc_neurocore.neurons.model_identity`: one row per registered class and
alias with its identity kind, count membership, taxonomy, schema profiles,
source locator, public fidelity status, revalidation status and the evidence
gates its descriptor does not yet claim, plus the derived catalogue counts and
the count definition. It carries no timestamps or commit hashes so that it only
changes when the catalogue changes.

Usage::

    python tools/model_identity_ledger.py --write
    python tools/model_identity_ledger.py --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sc_neurocore.neurons.model_identity import (  # noqa: E402
    COUNT_DEFINITION,
    NETWORK_IDENTITIES,
    catalogue_counts,
    identity_registry,
)

LEDGER_SCHEMA = "sc-neurocore.model-identity-ledger.v1"
DEFAULT_OUTPUT = Path("docs/_generated/model_identity_ledger.json")


def build_ledger() -> dict[str, object]:
    """Return the ledger payload derived from the live identity registry.

    Returns
    -------
    dict[str, object]
        JSON-compatible ledger with counts, definition, identities and networks.
    """
    registry = identity_registry()
    counts = catalogue_counts()
    identities = [record.to_public_dict() for record in registry.values()]
    schema_profiles = {
        profile.stem: record.class_name
        for record in registry.values()
        for profile in record.schema_profiles
    }
    return {
        "schema": LEDGER_SCHEMA,
        "count_definition": COUNT_DEFINITION,
        "counts": counts.to_public_dict(),
        "network_identities": [
            {
                "class_name": network.class_name,
                "module": network.module,
                "kind": network.kind,
                "cell_identity": network.cell_identity,
            }
            for network in NETWORK_IDENTITIES
        ],
        "schema_profiles": dict(sorted(schema_profiles.items())),
        "identities": identities,
    }


def render_ledger() -> str:
    """Return the ledger serialised exactly as it is written to disk."""
    return json.dumps(build_ledger(), indent=2, ensure_ascii=False, sort_keys=False) + "\n"


def ledger_problems(repo: Path, output: Path = DEFAULT_OUTPUT) -> list[str]:
    """Return the reasons the tracked ledger is missing or stale.

    Parameters
    ----------
    repo:
        Repository root.
    output:
        Ledger path relative to ``repo``.

    Returns
    -------
    list[str]
        Empty when the tracked ledger equals the freshly rendered one.
    """
    path = repo / output
    if not path.is_file():
        return [f"missing generated ledger: {output.as_posix()}"]
    if path.read_text(encoding="utf-8") != render_ledger():
        return [
            f"stale generated ledger: {output.as_posix()} (run tools/model_identity_ledger.py --write)"
        ]
    return []


def write_ledger(repo: Path, output: Path = DEFAULT_OUTPUT) -> Path:
    """Write the ledger and return its path."""
    path = repo / output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_ledger(), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="write the ledger")
    mode.add_argument("--check", action="store_true", help="fail if the tracked ledger is stale")
    mode.add_argument("--counts", action="store_true", help="print the derived catalogue counts")
    args = parser.parse_args(argv)
    if args.counts:
        print(json.dumps(catalogue_counts().to_public_dict(), indent=2))
        return 0
    if args.check:
        problems = ledger_problems(args.repo, args.output)
        for problem in problems:
            print(problem)
        return 1 if problems else 0
    path = write_ledger(args.repo, args.output)
    counts = catalogue_counts()
    print(
        f"wrote {path.relative_to(args.repo).as_posix()}: {counts.registered} classes, "
        f"{counts.source_catalogue} source-catalogue identities, "
        f"{counts.polyglot_complete_source} polyglot-complete"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

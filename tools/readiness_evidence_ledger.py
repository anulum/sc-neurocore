#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Generated readiness evidence ledger (declared versus verified)

"""Emit or check the generated readiness evidence ledger.

The ledger (``docs/_generated/readiness_evidence_ledger.json``) is the exact
claim ledger of :mod:`sc_neurocore.neurons.readiness`: for every registered
class it lists the declared science/silicon tiers (tier semantics v1), the
verified tiers (bound facet receipts only) and, per facet, the status, the
parsed evidence references with their resolution, the newest receipt and the
subjects that changed since it was recorded. It also carries the facet
definitions and the invalidation matrix so a reader can audit why a facet
holds its status. It records no timestamps or commit hashes; it changes only
when a descriptor, an evidence file, a receipt or a receipt subject changes.

Usage::

    python tools/readiness_evidence_ledger.py --write
    python tools/readiness_evidence_ledger.py --check
    python tools/readiness_evidence_ledger.py --summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sc_neurocore.neurons.facet_receipts import FACETS, INVALIDATION_MATRIX  # noqa: E402
from sc_neurocore.neurons.readiness import (  # noqa: E402
    FACET_STATUSES,
    readiness_report,
    summarise,
)

LEDGER_SCHEMA = "sc-neurocore.readiness-evidence-ledger.v1"
DEFAULT_OUTPUT = Path("docs/_generated/readiness_evidence_ledger.json")

STATUS_DEFINITION = {
    "not-declared": "the descriptor does not claim the facet",
    "declared": "claimed; the evidence field names nothing that can be located",
    "unavailable": "claimed; at least one named file or test node does not exist",
    "located": "every named file and test node exists; no receipt records a run",
    "bound": "the newest receipt is creditable and every subject digest still matches",
    "stale": "the newest receipt was creditable, but a subject changed or vanished",
    "invalid": "the newest receipt cannot credit the facet",
}


def build_ledger(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    """Return the ledger payload derived from the live verifier."""
    records = readiness_report(repo_root=repo_root)
    return {
        "schema": LEDGER_SCHEMA,
        "tier_semantics": {
            "declared": "descriptor_tiers v1: boolean anchor plus non-empty evidence string",
            "verified": "bound facet receipts only, one rung at a time over declared facets",
        },
        "status_definition": {status: STATUS_DEFINITION[status] for status in FACET_STATUSES},
        "facets": [
            {
                "name": spec.name,
                "axis": spec.axis,
                "rung": spec.rung,
                "required_subjects": list(spec.required_subjects),
                "optional_subjects": list(spec.optional_subjects),
                "evidence_field": spec.evidence_field,
                "claim_scope": spec.claim_scope,
            }
            for spec in FACETS
        ],
        "invalidation_matrix": {name: list(kinds) for name, kinds in INVALIDATION_MATRIX.items()},
        "summary": summarise(records.values()),
        "models": [record.to_public_dict() for record in records.values()],
    }


def render_ledger(repo_root: Path = REPO_ROOT) -> str:
    """Return the ledger serialised exactly as it is written to disk."""
    return json.dumps(build_ledger(repo_root), indent=2, ensure_ascii=False) + "\n"


def ledger_problems(repo: Path, output: Path = DEFAULT_OUTPUT) -> list[str]:
    """Return the reasons the tracked ledger is missing or stale."""
    path = repo / output
    if not path.is_file():
        return [f"missing generated ledger: {output.as_posix()}"]
    if path.read_text(encoding="utf-8") != render_ledger(repo):
        return [
            f"stale generated ledger: {output.as_posix()} "
            "(run tools/readiness_evidence_ledger.py --write)"
        ]
    return []


def write_ledger(repo: Path, output: Path = DEFAULT_OUTPUT) -> Path:
    """Write the ledger and return its path."""
    path = repo / output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_ledger(repo), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="write the ledger")
    mode.add_argument("--check", action="store_true", help="fail if the tracked ledger is stale")
    mode.add_argument("--summary", action="store_true", help="print the readiness summary")
    args = parser.parse_args(argv)
    if args.summary:
        print(json.dumps(summarise(readiness_report(repo_root=args.repo).values()), indent=2))
        return 0
    if args.check:
        problems = ledger_problems(args.repo, args.output)
        for problem in problems:
            print(problem)
        return 1 if problems else 0
    path = write_ledger(args.repo, args.output)
    summary = summarise(readiness_report(repo_root=args.repo).values())
    print(
        f"wrote {path.relative_to(args.repo).as_posix()}: {summary['models']} models, "
        f"verified silicon {summary['verified_silicon_tiers']}, "
        f"verified science {summary['verified_science_tiers']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

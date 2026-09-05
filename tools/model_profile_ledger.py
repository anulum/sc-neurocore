#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Generated model profile ledger (scientific / numerical / lowering)

"""Emit or check the generated model profile ledger.

The ledger (``docs/_generated/model_profile_ledger.json``) is the inventory
:mod:`sc_neurocore.neurons.profile_registry` derives: one row per bound
(class, schema profile) with the separated scientific model, numerical
realisation and lowering profile, the descriptor's own integration label next
to the profile's method, the per-profile validator registry with each
validator's resolution and receipt status, and the admission verdict. It also
carries the contract identifier and the method mapping table. It records no
timestamps or commit hashes; it changes only when a schema, a descriptor, an
evidence file, a receipt or a receipt subject changes.

Usage::

    python tools/model_profile_ledger.py --write
    python tools/model_profile_ledger.py --check
    python tools/model_profile_ledger.py --summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sc_neurocore.neurons.model_profile import (  # noqa: E402
    EXECUTABLE_DETECTIONS,
    EXECUTABLE_METHODS,
    PROFILE_CONTRACT,
)
from sc_neurocore.neurons.profile_registry import (  # noqa: E402
    method_table,
    profile_inventory,
    summarise_inventory,
)

LEDGER_SCHEMA = "sc-neurocore.model-profile-ledger.v1"
DEFAULT_OUTPUT = Path("docs/_generated/model_profile_ledger.json")

ADMISSION_DEFINITION = {
    "admitted": (
        "executable, contradiction-free, and every declared facet has at least one "
        "executable validator"
    ),
    "blocked": (
        "executable, but the profile contradicts its schema or a declared facet is backed "
        "by prose alone"
    ),
    "not-executable": (
        "a descriptive record whose method or detection is outside the executable vocabulary"
    ),
}


def build_ledger(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    """Return the ledger payload derived from the live registry."""
    rows = profile_inventory(repo_root=repo_root)
    return {
        "schema": LEDGER_SCHEMA,
        "contract": PROFILE_CONTRACT,
        "executable_vocabulary": {
            "methods": list(EXECUTABLE_METHODS),
            "detections": list(EXECUTABLE_DETECTIONS),
        },
        "method_table": list(method_table()),
        "admission_definition": dict(ADMISSION_DEFINITION),
        "summary": summarise_inventory(rows),
        "profiles": [row.to_public_dict() for row in rows],
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
            f"stale generated ledger: {output.as_posix()} (run tools/model_profile_ledger.py --write)"
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
    mode.add_argument("--summary", action="store_true", help="print the inventory summary")
    args = parser.parse_args(argv)
    if args.summary:
        print(json.dumps(summarise_inventory(profile_inventory(repo_root=args.repo)), indent=2))
        return 0
    if args.check:
        problems = ledger_problems(args.repo, args.output)
        for problem in problems:
            print(problem)
        return 1 if problems else 0
    path = write_ledger(args.repo, args.output)
    summary = summarise_inventory(profile_inventory(repo_root=args.repo))
    print(
        f"wrote {path.relative_to(args.repo).as_posix()}: {summary['profiles']} profiles over "
        f"{summary['classes']} classes, admission {summary['admission']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

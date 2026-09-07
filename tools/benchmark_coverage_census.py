#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Which catalogue models have a benchmark measured

"""How many catalogue models have a benchmark measured, counted rather than asserted.

The 7-point audit records "Benchmarks FAIL — 63/173 covered (36%)" as a
hand-written row. Nothing computes it. Two live measurements disagree with that
figure and with each other, and the difference between them is the finding:

* 101 of 185 models are *named by* a benchmark script;
* 26 of 185 have a *committed benchmark record* naming them.

A script that exists is apparatus, not a measurement. The 75 models between the
two figures have the machinery to be benchmarked and no evidence that anyone
ran it, which is precisely the gap a coverage number is supposed to expose.

This census separates the three states so the distinction cannot be lost again:

``measured``
    a committed record under ``benchmarks/results`` names the model.
``apparatus_only``
    a benchmark script names the model; no committed record does.
``absent``
    neither.

It reads only committed sources, records no timestamps, host names or commit
hashes, and answers nothing about whether a lane is built here today —
`tools/benchmark_evidence_gate.py` judges the *quality* of the artefacts that
exist, and this counts *how many models have any*. The two are different
questions and neither substitutes for the other.

Usage::

    python tools/benchmark_coverage_census.py --write
    python tools/benchmark_coverage_census.py --check
    python tools/benchmark_coverage_census.py --summary
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sc_neurocore.neurons.models import _CLASS_TO_MODULE  # noqa: E402

#: Schema of the emitted census.
CENSUS_SCHEMA = "sc-neurocore.benchmark-coverage-census.v1"

#: Where the census is committed.
DEFAULT_OUTPUT = "docs/_generated/benchmark_coverage_census.json"

#: Directory holding committed benchmark evidence.
RESULTS_DIR = "benchmarks/results"

#: Directory holding the benchmark scripts.
SCRIPTS_DIR = "benchmarks"

#: The three states a catalogue model can be in.
STATES = ("measured", "apparatus_only", "absent")


def model_names() -> tuple[str, ...]:
    """Return every registered catalogue model class name, sorted."""
    return tuple(sorted(_CLASS_TO_MODULE))


def _names_in_payload(payload: object, wanted: frozenset[str]) -> set[str]:
    """Return every wanted model name appearing as a string anywhere in *payload*.

    A record may name its model at the top level, inside a nested ``meta`` or
    ``verification`` block, or only in a ``kernel`` label. Scanning every string
    avoids counting a record as evidence-free merely because its schema differs
    from the one the scanner expected.
    """
    found: set[str] = set()
    if isinstance(payload, str):
        if payload in wanted:
            found.add(payload)
    elif isinstance(payload, dict):
        for value in payload.values():
            found |= _names_in_payload(value, wanted)
    elif isinstance(payload, list):
        for value in payload:
            found |= _names_in_payload(value, wanted)
    return found


def measured_models(repo_root: Path = REPO_ROOT) -> dict[str, list[str]]:
    """Return each model that a committed benchmark record names, with the records.

    Parameters
    ----------
    repo_root : Path
        Repository root; the results directory is resolved beneath it.

    Returns
    -------
    dict
        Model name to the sorted record filenames naming it. A model absent
        from the mapping has no committed benchmark evidence at all.
    """
    wanted = frozenset(model_names())
    found: dict[str, set[str]] = {}
    for path in sorted((repo_root / RESULTS_DIR).rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for name in _names_in_payload(payload, wanted):
            found.setdefault(name, set()).add(path.name)
    return {name: sorted(records) for name, records in sorted(found.items())}


def apparatus_models(repo_root: Path = REPO_ROOT) -> dict[str, list[str]]:
    """Return each model a benchmark script names, with the scripts naming it.

    Parameters
    ----------
    repo_root : Path
        Repository root; the scripts directory is resolved beneath it.

    Returns
    -------
    dict
        Model name to the sorted script filenames naming it.
    """
    wanted = model_names()
    found: dict[str, set[str]] = {}
    for path in sorted((repo_root / SCRIPTS_DIR).glob("*.py")):
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for name in wanted:
            if name in source:
                found.setdefault(name, set()).add(path.name)
    return {name: sorted(scripts) for name, scripts in sorted(found.items())}


def build_census(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    """Return the coverage census over the whole catalogue.

    Parameters
    ----------
    repo_root : Path
        Repository root to read committed sources from.

    Returns
    -------
    dict
        The census payload: one row per model with its state and bindings,
        plus totals per state.
    """
    measured = measured_models(repo_root)
    apparatus = apparatus_models(repo_root)
    rows: list[dict[str, object]] = []
    for name in model_names():
        if name in measured:
            state = "measured"
        elif name in apparatus:
            state = "apparatus_only"
        else:
            state = "absent"
        rows.append(
            {
                "model": name,
                "state": state,
                "records": measured.get(name, []),
                "scripts": apparatus.get(name, []),
            }
        )
    totals = {state: sum(1 for row in rows if row["state"] == state) for state in STATES}
    return {
        "schema_version": CENSUS_SCHEMA,
        "catalogue_models": len(rows),
        "totals": totals,
        "rows": rows,
    }


def encode(census: dict[str, object]) -> str:
    """Return the census as the exact bytes committed to disk."""
    return json.dumps(census, indent=2, sort_keys=True) + "\n"


def render_summary(census: dict[str, object]) -> str:
    """Return a short operator-readable summary of the census."""
    totals = census["totals"]
    assert isinstance(totals, dict)
    total = census["catalogue_models"]
    assert isinstance(total, int)
    lines = [f"catalogue models: {total}"]
    for state in STATES:
        count = int(totals[state])
        share = (100 * count) // total if total else 0
        lines.append(f"  {state:15s} {count:4d}  ({share}%)")
    lines.append("a benchmark script that names a model is apparatus, not a measurement")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Write, check or summarise the census from the command line.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments; ``sys.argv[1:]`` when omitted.

    Returns
    -------
    int
        ``0`` on success, ``1`` when ``--check`` finds the file absent or stale.
    """
    parser = argparse.ArgumentParser(
        description="Count catalogue models with a benchmark measured."
    )
    parser.add_argument("--write", action="store_true", help="write the census")
    parser.add_argument("--check", action="store_true", help="fail when the file has drifted")
    parser.add_argument("--summary", action="store_true", help="print the census summary")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)

    census = build_census()
    encoded = encode(census)
    if arguments.check:
        if not arguments.output.is_file():
            print(f"{arguments.output} is absent; run --write.", file=sys.stderr)
            return 1
        if arguments.output.read_text(encoding="utf-8") != encoded:
            print(f"{arguments.output} has drifted; run --write.", file=sys.stderr)
            return 1
        print(f"{arguments.output} is up to date")
        return 0
    if arguments.write:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded, encoding="utf-8")
        print(f"Wrote {arguments.output}")
    if arguments.summary or not (arguments.write or arguments.check):
        print(render_summary(census))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

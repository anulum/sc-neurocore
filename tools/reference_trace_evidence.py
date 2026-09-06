#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Reference-trace independence and control evidence

"""Print what the reference-trace corpus actually establishes.

Two questions an operator should be able to answer without reading code: which
identities rest on an externally published source rather than on a formulation
this repository retains, and whether each trace can fail at all when the model
is broken.

``--check`` is the fail-closed form: it exits non-zero when a trace cannot be
adjudicated, or when some trace survives every negative control and therefore
proves nothing.

Usage::

    PYTHONPATH=src:. python tools/reference_trace_evidence.py --report
    PYTHONPATH=src:. python tools/reference_trace_evidence.py --check
    PYTHONPATH=src:. python tools/reference_trace_evidence.py --json out.json
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys

from sc_neurocore.neurons.reference_trace_mutations import negative_control_report
from sc_neurocore.neurons.reference_trace_provenance import (
    ReferenceTraceAdjudicationError,
    adjudicate_corpus,
)


def build_evidence() -> dict[str, object]:
    """Return the adjudication and negative-control evidence together.

    Returns
    -------
    dict
        ``{"adjudication": ..., "negative_controls": ...}``, both JSON-safe.
    """
    return {
        "adjudication": adjudicate_corpus().to_public_dict(),
        "negative_controls": negative_control_report().to_public_dict(),
    }


def render_report(evidence: dict[str, object]) -> str:
    """Render the evidence as operator-readable lines.

    Parameters
    ----------
    evidence : dict
        Output of :func:`build_evidence`.

    Returns
    -------
    str
        A short report: independence counts, then every control outcome that is
        not a plain detection.
    """
    adjudication = _mapping(evidence["adjudication"])
    controls = _mapping(evidence["negative_controls"])
    lines = [f"independence adjudication ({adjudication['version']})"]
    counts = _mapping(adjudication["counts"])
    for name in sorted(counts):
        lines.append(f"  {counts[name]:>3}  {name}")
    lines.append("")
    lines.append(f"negative controls ({controls['version']})")
    outcomes = [_mapping(row) for row in _sequence(controls["outcomes"])]
    tally: dict[str, int] = {}
    for outcome in outcomes:
        tally[str(outcome["status"])] = tally.get(str(outcome["status"]), 0) + 1
    for status in sorted(tally):
        lines.append(f"  {tally[status]:>3}  {status}")
    uncontrolled = [str(name) for name in _sequence(controls["uncontrolled_traces"])]
    lines.append("")
    lines.append(
        "  every trace fails under at least one control"
        if not uncontrolled
        else f"  UNCONTROLLED: {', '.join(uncontrolled)}"
    )
    undetected = [row for row in outcomes if row["status"] == "undetected"]
    if undetected:
        lines.append("")
        lines.append("  applied but undetected:")
        for row in sorted(undetected, key=lambda item: (str(item["name"]), str(item["mutation"]))):
            lines.append(f"    {row['name']} / {row['mutation']}: {row['reason']}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the evidence tool.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments; defaults to :data:`sys.argv`.

    Returns
    -------
    int
        ``0`` when the corpus adjudicates and every trace is controlled, ``1``
        otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", help="print the readable report")
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero when a trace cannot be adjudicated or is uncontrolled",
    )
    parser.add_argument("--json", type=Path, default=None, help="write the evidence as JSON")
    arguments = parser.parse_args(argv)

    try:
        evidence = build_evidence()
    except ReferenceTraceAdjudicationError as error:
        print(f"reference-trace evidence unavailable: {error}", file=sys.stderr)
        return 1

    if arguments.json is not None:
        arguments.json.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if arguments.report or not (arguments.check or arguments.json):
        print(render_report(evidence))

    uncontrolled = _sequence(_mapping(evidence["negative_controls"])["uncontrolled_traces"])
    if arguments.check and uncontrolled:
        print(
            "uncontrolled reference traces prove nothing: "
            + ", ".join(str(n) for n in uncontrolled),
            file=sys.stderr,
        )
        return 1
    return 0


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"expected a mapping, got {type(value).__name__}")
    return {str(key): item for key, item in value.items()}


def _sequence(value: object) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"expected a list, got {type(value).__name__}")
    return list(value)


if __name__ == "__main__":  # pragma: no cover - console entry point
    raise SystemExit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Generated foreign runtime state conformance matrix

"""Emit or check how much of each model's state a foreign runtime can carry.

The matrix (``docs/_generated/runtime_state_conformance.json``) answers one
question per model and lane: of the state variables this model declares, which
could that lane transport, which would it drop, and does it export anything the
model has no name for. It is derived from the declared layouts and the lane
packets (``sc-neurocore.runtime-state-packet.v1``), so it is reproducible from
committed sources and records no timestamps or commit hashes.

What it deliberately does not answer: whether a lane is built, installed or able
to run a given model today. That is a property of a machine, not of the
contract, and it is what ``tests/test_rust_python_neuron_parity.py`` checks. A
matrix that mixed the two would change with the toolchain on the box.

Usage::

    python tools/runtime_state_conformance.py --write
    python tools/runtime_state_conformance.py --check
    python tools/runtime_state_conformance.py --summary
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from typing import Any
import json
from pathlib import Path
import re
import sys

try:  # pragma: no cover - covered by the Python-version matrix.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sc_neurocore.studio.model_catalogue import list_models  # noqa: E402
from sc_neurocore.studio.runtime_state_packet import (  # noqa: E402
    RUNTIME_STATE_PACKET_SCHEMA_VERSION,
    RUST_BATCH_PACKET,
    RuntimeStatePacket,
    packet_coverage,
)
from sc_neurocore.studio.state_layout import declared_state  # noqa: E402

CONFORMANCE_SCHEMA = "sc-neurocore.runtime-state-conformance.v1"
DEFAULT_CEILING = Path("tools/runtime_state_ceiling.toml")
DEFAULT_OUTPUT = Path("docs/_generated/runtime_state_conformance.json")

#: The committed source that names every model the native network runner can be
#: selected for. It is read as text rather than through the built engine so the
#: matrix stays derivable from committed files on a machine with no engine.
LANE_CATALOGUE_SOURCE = Path("engine/src/network_runner/model_catalogue.rs")

_CATALOGUE_NAME = re.compile(r'^\s*"([A-Za-z0-9_]+)"\s*,?\s*$')

#: Every lane with a declared transport packet. A lane absent from this tuple
#: has no packet yet, which the matrix says rather than implying full transport.
LANES: tuple[RuntimeStatePacket, ...] = (RUST_BATCH_PACKET,)


def model_names() -> tuple[str, ...]:
    """Return the catalogue model names, sorted."""
    names = {str(entry["name"]) for entry in list_models()}
    return tuple(sorted(names))


def selectable_models(repo_root: Path = REPO_ROOT) -> frozenset[str]:
    """Return the model names the native network runner can be selected for.

    Parsed from the committed Rust catalogue rather than asked of the built
    engine, so the answer is the same on a machine that has never compiled it.
    A catalogue entry matches a Python identity either exactly or with the
    ``Neuron`` suffix removed, which is the rule the dispatcher itself applies.
    """
    text = (repo_root / LANE_CATALOGUE_SOURCE).read_text(encoding="utf-8")
    body = text.split("vec![", 1)[1].split("]", 1)[0] if "vec![" in text else ""
    return frozenset(
        match.group(1)
        for match in (_CATALOGUE_NAME.match(line) for line in body.splitlines())
        if match
    )


def _is_selectable(model: str, catalogue: frozenset[str]) -> bool:
    """Return whether the lane can be selected for a Python model identity."""
    return model in catalogue or (model.endswith("Neuron") and model[:-6] in catalogue)


def build_matrix() -> dict[str, object]:
    """Return the conformance matrix for every catalogue model and lane.

    Returns
    -------
    dict
        ``lanes``, one ``rows`` entry per model, and a ``summary`` census.
    """
    rows: list[dict[str, object]] = []
    census = {
        packet.runtime: {"carried": 0, "complete": 0, "dropped": 0, "names_nothing": 0}
        for packet in LANES
    }
    # The same counts restricted to the models each lane can be selected for,
    # which is the question an operator asks about a run rather than about the
    # catalogue. Emitted here so one drift gate holds both censuses.
    selectable_census = {
        packet.runtime: {
            "carried": 0,
            "complete": 0,
            "dropped": 0,
            "models": 0,
            "models_with_declared_state": 0,
            "names_nothing": 0,
        }
        for packet in LANES
    }
    catalogue = selectable_models()
    undeclared = 0
    for name in model_names():
        source, _profile, declared = declared_state(name)
        declared_names = tuple(spec.name for spec in declared)
        if not declared_names:
            undeclared += 1
        lanes: dict[str, object] = {}
        for packet in LANES:
            coverage = packet_coverage(packet, declared_names)
            lanes[packet.runtime] = coverage.to_public_dict()
            counts = census[packet.runtime]
            counts["carried"] += len(coverage.carried)
            counts["dropped"] += len(coverage.dropped)
            if declared_names and coverage.complete:
                counts["complete"] += 1
            if declared_names and coverage.names_nothing:
                counts["names_nothing"] += 1
            if _is_selectable(name, catalogue):
                restricted = selectable_census[packet.runtime]
                restricted["models"] += 1
                if declared_names:
                    restricted["models_with_declared_state"] += 1
                    restricted["carried"] += len(coverage.carried)
                    restricted["dropped"] += len(coverage.dropped)
                    if coverage.complete:
                        restricted["complete"] += 1
                    if coverage.names_nothing:
                        restricted["names_nothing"] += 1
        rows.append(
            {
                "declared": list(declared_names),
                "layout_source": source,
                "lanes": lanes,
                "model": name,
            }
        )
    return {
        "lanes": [packet.to_public_dict() for packet in LANES],
        "packet_schema_version": RUNTIME_STATE_PACKET_SCHEMA_VERSION,
        "rows": rows,
        "schema_version": CONFORMANCE_SCHEMA,
        "summary": {
            "models": len(rows),
            "models_without_declared_state": undeclared,
            "per_lane": {
                runtime: dict(sorted(counts.items())) for runtime, counts in census.items()
            },
            "per_lane_selectable": {
                runtime: dict(sorted(counts.items()))
                for runtime, counts in selectable_census.items()
            },
        },
    }


def render_summary(matrix: dict[str, object]) -> str:
    """Return the operator-readable census of a matrix.

    Parameters
    ----------
    matrix : dict
        The result of :func:`build_matrix`.

    Returns
    -------
    str
        One block per lane: how many models it fully accounts for, how many it
        can name nothing in, and how many declared variables it carries and
        drops — first across the whole catalogue, then restricted to the models
        that lane can actually be selected for, which is the question an
        operator asks about a run rather than about the catalogue.
    """
    summary = matrix["summary"]
    if not isinstance(summary, dict):
        raise TypeError("A conformance matrix expected a mapping for its summary.")
    per_lane = summary["per_lane"]
    if not isinstance(per_lane, dict):
        raise TypeError("A conformance matrix expected a mapping for its lane census.")
    selectable = summary.get("per_lane_selectable", {})
    if not isinstance(selectable, dict):
        raise TypeError("A conformance matrix expected a mapping for its selectable census.")
    lines = [
        f"foreign runtime state conformance ({matrix['schema_version']})",
        f"  catalogue models: {summary['models']}"
        f" ({summary['models_without_declared_state']} declare no state)",
    ]
    for runtime, counts in sorted(per_lane.items()):
        if not isinstance(counts, dict):  # pragma: no cover - built by this module
            raise TypeError("A conformance matrix expected a mapping for each lane.")
        lines.append(f"  {runtime}:")
        lines.append(f"    declared variables carried: {counts['carried']}")
        lines.append(f"    declared variables dropped: {counts['dropped']}")
        lines.append(f"    models fully accounted for: {counts['complete']}")
        lines.append(f"    models it can name nothing in: {counts['names_nothing']}")
        restricted = selectable.get(runtime)
        if isinstance(restricted, dict):
            lines.append(
                f"    of the {restricted['models']} it can be selected for,"
                f" {restricted['models_with_declared_state']} declare a layout:"
            )
            lines.append(f"      declared variables carried: {restricted['carried']}")
            lines.append(f"      declared variables dropped: {restricted['dropped']}")
            lines.append(f"      models fully accounted for: {restricted['complete']}")
            lines.append(f"      models it can name nothing in: {restricted['names_nothing']}")
    return "\n".join(lines)


def encode(matrix: dict[str, object]) -> str:
    """Return the matrix as the file records it."""
    return json.dumps(matrix, indent=2, sort_keys=True) + "\n"


class RuntimeStateCeilingError(RuntimeError):
    """Raised when a ceiling file cannot be read or would be loosened."""


def ceiling_verdicts(matrix: Mapping[str, Any], ceiling: Mapping[str, Any]) -> list[str]:
    """Return one line per lane whose transport is worse than its ceiling.

    FF-05 is a per-model, per-lane campaign that will run for a long time. The
    matrix already refuses to drift from what it derives, but regenerating it
    accepts a worse number as readily as a better one. These ceilings make the
    direction part of the contract: dropped state may only fall, complete
    models may only rise.

    Parameters
    ----------
    matrix : Mapping[str, Any]
        A matrix as :func:`build_matrix` returns it.
    ceiling : Mapping[str, Any]
        Parsed ceiling document, keyed by lane runtime.

    Returns
    -------
    list of str
        Human-readable regressions, empty when every lane is at or better than
        its ceiling.
    """
    summary = matrix["summary"]
    assert isinstance(summary, Mapping)
    per_lane = summary["per_lane"]
    assert isinstance(per_lane, Mapping)
    lanes = ceiling.get("lanes", {})
    if not isinstance(lanes, Mapping):
        raise RuntimeStateCeilingError("ceiling 'lanes' must be a table")

    verdicts: list[str] = []
    for runtime, counts in sorted(per_lane.items()):
        limits = lanes.get(runtime)
        if limits is None:
            verdicts.append(f"{runtime}: no ceiling recorded; run --update-ceiling")
            continue
        dropped = int(counts["dropped"])
        complete = int(counts["complete"])
        if dropped > int(limits["max_dropped"]):
            verdicts.append(
                f"{runtime}: {dropped} state variables dropped, ceiling "
                f"{limits['max_dropped']} — transport got worse"
            )
        if complete < int(limits["min_complete"]):
            verdicts.append(
                f"{runtime}: {complete} models fully carried, floor "
                f"{limits['min_complete']} — transport got worse"
            )
    return verdicts


def render_ceiling(matrix: Mapping[str, Any]) -> str:
    """Return the ceiling document the current matrix justifies."""
    summary = matrix["summary"]
    assert isinstance(summary, Mapping)
    per_lane = summary["per_lane"]
    assert isinstance(per_lane, Mapping)
    lines = [
        "# SPDX-" + "License-Identifier: AGPL-3.0-or-later",
        "# Commercial license available",
        "# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.",
        "# © Code 2020–2026 Miroslav Šotek. All rights reserved.",
        "# ORCID: 0009-0009-3560-0851",
        "# Contact: www.anulum.li | protoscience@anulum.li",
        "# SC-NeuroCore — Runtime state transport ratchets (FF-05)",
        "",
        "# Measured ceilings, never copied: `--update-ceiling` writes what the",
        "# matrix currently derives and refuses to loosen one. Dropped state may",
        "# only fall; fully carried models may only rise. A lane absent here has",
        "# no ceiling yet and the guard says so rather than passing it.",
        "",
        "schema_version = 1",
        "",
    ]
    for runtime, counts in sorted(per_lane.items()):
        lines.append(f'[lanes."{runtime}"]')
        lines.append(f"max_dropped = {int(counts['dropped'])}")
        lines.append(f"min_complete = {int(counts['complete'])}")
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Write, check or summarise the matrix from the command line.

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
        description="Emit or check the foreign runtime state conformance matrix."
    )
    parser.add_argument("--write", action="store_true", help="write the matrix")
    parser.add_argument("--check", action="store_true", help="fail when the file has drifted")
    parser.add_argument("--summary", action="store_true", help="print the census")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / DEFAULT_OUTPUT)
    parser.add_argument(
        "--check-ceiling",
        action="store_true",
        help="fail when a lane transports less state than its recorded ceiling",
    )
    parser.add_argument(
        "--update-ceiling",
        action="store_true",
        help="tighten the ceilings to what the matrix now derives; never loosens",
    )
    parser.add_argument("--ceiling", type=Path, default=REPO_ROOT / DEFAULT_CEILING)
    arguments = parser.parse_args(argv)

    matrix = build_matrix()
    encoded = encode(matrix)
    if arguments.check_ceiling or arguments.update_ceiling:
        return _run_ceiling(matrix, arguments)
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
        print(render_summary(matrix))
    return 0


def _run_ceiling(matrix: Mapping[str, Any], arguments: argparse.Namespace) -> int:
    """Check or tighten the transport ceilings; never loosen one."""
    path = arguments.ceiling
    if arguments.check_ceiling:
        if not path.is_file():
            print(f"{path} is absent; run --update-ceiling.", file=sys.stderr)
            return 1
        ceiling = tomllib.loads(path.read_text(encoding="utf-8"))
        verdicts = ceiling_verdicts(matrix, ceiling)
        if verdicts:
            for line in verdicts:
                print(line, file=sys.stderr)
            return 1
        print(f"{path}: every lane is at or better than its ceiling")
        return 0

    proposed = render_ceiling(matrix)
    if path.is_file():
        existing = tomllib.loads(path.read_text(encoding="utf-8"))
        loosened = ceiling_verdicts(matrix, existing)
        if loosened:
            for line in loosened:
                print(f"refusing to loosen: {line}", file=sys.stderr)
            return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(proposed, encoding="utf-8")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - command-line entry point
    raise SystemExit(main())

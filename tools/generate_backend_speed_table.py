# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Packaged measured backend order generator

"""Generate the measured backend order the installed package reads, from current or reviewed records only.

The polyglot dispatch order of a kernel may follow a measurement only while that measurement still describes the
current source. This tool reads every tracked benchmark record under ``benchmarks/results`` that has the selector's
shape (a ``kernel`` id, a ``meta.cpu`` host model and a ``backends`` map of per-language timings), asks the coverage
census for its currency with the census's own inputs, and keeps a record only when its currency is ``current`` or
``reviewed``. The result is written to ``src/sc_neurocore/accel/measured_backend_orders.json``, which the selector
reads through the package's resources, so a source checkout and an installed wheel order lanes the same way.

Every record that is left out is listed with its reason, never dropped silently:

``shape``
    The file is not a selector-shaped record (it lacks ``kernel``, ``meta.cpu`` or ``backends``, or does not parse).
``outdated``, ``unbound``
    The census's currency: a bound source has changed since the measurement, or the record binds no source.
``no_timed_backend``
    No backend in the record ran with a numeric median.
``duplicate``
    Two or more admitted records measure the same kernel on the same host model and level; none is chosen by guess.

An entry's ``isa_level`` is the x86-64 psABI level of the measuring host, taken from the record's ``meta.cpu_flags``
when it records them and ``"unrecorded"`` otherwise. The selector applies an unrecorded entry only on an exact host
model match and a recorded one only on a host at that level or above.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
from typing import TypedDict

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from sc_neurocore.accel.host_isa import level_from_flags
from tools.benchmark_coverage_census import RESULTS_DIR, manifest_source_paths, record_currency
from tools.benchmark_evidence_gate import tracked_files
from tools.source_change_reviews import load_reviews

#: Schema of the generated table.
TABLE_SCHEMA = "sc-neurocore.measured-backend-orders.v1"

#: Where the table is committed; it ships inside the package.
DEFAULT_OUTPUT = "src/sc_neurocore/accel/measured_backend_orders.json"

#: The census currencies that admit a record.
ADMITTED = ("current", "reviewed")

#: The level given to a record that does not state its host's CPU flags.
UNRECORDED = "unrecorded"


class Entry(TypedDict):
    """One admitted measurement: the order of the lanes that ran, fastest first."""

    kernel: str
    cpu_model: str
    isa_level: str
    order: list[str]
    record: str
    record_sha256: str
    source_digests: dict[str, str]
    currency: str


class LeftOut(TypedDict):
    """A record that orders nothing, with the reason."""

    record: str
    reason: str


class Table(TypedDict):
    """The whole generated table."""

    schema_version: str
    entries: list[Entry]
    left_out: list[LeftOut]


def speed_order(backends: Mapping[str, object]) -> list[str]:
    """Return the backends that ran, fastest first by ``median_call_ms``.

    Parameters
    ----------
    backends : mapping
        The record's ``backends`` map; each value is the lane's result.

    Returns
    -------
    list of str
        The names of the lanes marked ``available`` and ``used`` with a numeric median, in ascending median order;
        ties keep the record's order.
    """
    timed: list[tuple[float, int, str]] = []
    for position, (name, entry) in enumerate(backends.items()):
        if not isinstance(entry, Mapping) or not (entry.get("available") and entry.get("used")):
            continue
        median = entry.get("median_call_ms")
        if isinstance(median, bool) or not isinstance(median, (int, float)):
            continue
        timed.append((float(median), position, name))
    timed.sort()
    return [name for _median, _position, name in timed]


def isa_level(meta: Mapping[str, object]) -> str:
    """Return the psABI level a record states for its host, or :data:`UNRECORDED`.

    Parameters
    ----------
    meta : mapping
        The record's ``meta`` block.

    Returns
    -------
    str
        The level from ``meta.cpu_flags`` when it is a list of flag names, else :data:`UNRECORDED`.
    """
    flags = meta.get("cpu_flags")
    if isinstance(flags, list) and all(isinstance(flag, str) for flag in flags):
        return level_from_flags(flags)
    return UNRECORDED


def _source_digests(payload: Mapping[str, object]) -> dict[str, str]:
    """Return the record's ``source_hashes`` when it maps source paths to digests, else an empty mapping."""
    hashes = payload.get("source_hashes")
    if isinstance(hashes, Mapping) and all(
        isinstance(path, str) and isinstance(digest, str) for path, digest in hashes.items()
    ):
        return dict(sorted(hashes.items()))
    return {}


def build(root: Path) -> Table:
    """Build the table from the tracked records of a work tree.

    Parameters
    ----------
    root : Path
        The repository root; it must be the top of a Git work tree.

    Returns
    -------
    Table
        The admitted entries, sorted by host model, level and kernel, and every record left out with its reason,
        sorted by record path.

    Raises
    ------
    ValueError
        When ``root`` is not the top of a Git work tree.
    """
    listed = tracked_files(root, RESULTS_DIR)
    if listed is None:
        raise ValueError(
            f"{root} is not the top of a Git work tree; only tracked records are read."
        )
    named = manifest_source_paths(root)
    reviews = load_reviews(root)
    candidates: list[Entry] = []
    left_out: list[LeftOut] = []
    prefix = RESULTS_DIR + "/"
    for path in sorted(listed):
        name = path.removeprefix(prefix)
        if "/" in name or not (name.startswith("bench_") and name.endswith(".json")):
            continue
        raw = (root / path).read_bytes()
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            left_out.append({"record": name, "reason": "shape"})
            continue
        meta = payload.get("meta") if isinstance(payload, dict) else None
        backends = payload.get("backends") if isinstance(payload, dict) else None
        kernel = payload.get("kernel") if isinstance(payload, dict) else None
        cpu = meta.get("cpu") if isinstance(meta, dict) else None
        if not (isinstance(kernel, str) and isinstance(cpu, str) and isinstance(backends, dict)):
            left_out.append({"record": name, "reason": "shape"})
            continue
        assert isinstance(meta, dict)
        currency = record_currency(name, root, named.get(name), reviews)["currency"]
        assert isinstance(currency, str)
        if currency not in ADMITTED:
            left_out.append({"record": name, "reason": currency})
            continue
        order = speed_order(backends)
        if not order:
            left_out.append({"record": name, "reason": "no_timed_backend"})
            continue
        candidates.append(
            {
                "kernel": kernel,
                "cpu_model": cpu,
                "isa_level": isa_level(meta),
                "order": order,
                "record": name,
                "record_sha256": hashlib.sha256(raw).hexdigest(),
                "source_digests": _source_digests(payload),
                "currency": currency,
            }
        )
    keys: dict[tuple[str, str, str], int] = {}
    for entry in candidates:
        key = (entry["cpu_model"], entry["isa_level"], entry["kernel"])
        keys[key] = keys.get(key, 0) + 1
    entries: list[Entry] = []
    for entry in candidates:
        if keys[(entry["cpu_model"], entry["isa_level"], entry["kernel"])] > 1:
            left_out.append({"record": entry["record"], "reason": "duplicate"})
        else:
            entries.append(entry)
    entries.sort(key=lambda entry: (entry["cpu_model"], entry["isa_level"], entry["kernel"]))
    left_out.sort(key=lambda item: item["record"])
    return {"schema_version": TABLE_SCHEMA, "entries": entries, "left_out": left_out}


def render(table: Table) -> str:
    """Return the table's committed text: sorted keys, two-space indent, one final newline.

    Parameters
    ----------
    table : Table
        The table to render.

    Returns
    -------
    str
        The JSON text.
    """
    return json.dumps(table, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def summary(table: Table) -> str:
    """Return a one-line count of the admitted entries and of the records left out, by reason.

    Parameters
    ----------
    table : Table
        The table to summarise.

    Returns
    -------
    str
        The counts.
    """
    reasons: dict[str, int] = {}
    for item in table["left_out"]:
        reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1
    counted = ", ".join(f"{reason} {count}" for reason, count in sorted(reasons.items()))
    return f"entries {len(table['entries'])}; left out {len(table['left_out'])} ({counted})"


def main(argv: Sequence[str] | None = None) -> int:
    """Write, check or summarise the table.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments; ``sys.argv[1:]`` when omitted.

    Returns
    -------
    int
        ``0`` on success, ``1`` when ``--check`` finds the file absent or stale, ``2`` when the root is not the top
        of a Git work tree.
    """
    parser = argparse.ArgumentParser(description="Generate the packaged measured backend order.")
    parser.add_argument("--write", action="store_true", help="write the table")
    parser.add_argument(
        "--check", action="store_true", help="fail when the committed table has drifted"
    )
    parser.add_argument("--summary", action="store_true", help="print the counts")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args(argv)
    root: Path = arguments.root
    output: Path = arguments.output if arguments.output is not None else root / DEFAULT_OUTPUT
    try:
        table = build(root)
    except ValueError as refused:
        print(refused, file=sys.stderr)
        return 2
    text = render(table)
    if arguments.write:
        output.write_text(text, encoding="utf-8")
    if arguments.check:
        if not output.is_file():
            print(f"{output} is absent; run --write.", file=sys.stderr)
            return 1
        if output.read_text(encoding="utf-8") != text:
            print(f"{output} has drifted; run --write.", file=sys.stderr)
            return 1
    if arguments.summary:
        print(summary(table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

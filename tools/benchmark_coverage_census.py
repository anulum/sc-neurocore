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
* 84 of 185 are measured by a *committed benchmark record*: 28 by a record that
  holds their class name, 56 more by a record that binds their benchmark script.

A script that exists is apparatus, not a measurement. The 17 models between the
two figures have the machinery to be benchmarked and no evidence that anyone
ran it, which is precisely the gap a coverage number is supposed to expose.

This census separates the three states so the distinction cannot be lost again:

``measured``
    a committed record under ``benchmarks/results`` measures the model: it
    holds the model's class name, or it binds by path benchmark scripts that
    between them name this model and no other. The record is the evidence in
    both cases; a script without a record stays apparatus.
``apparatus_only``
    a benchmark script names the model; no committed record does.
``absent``
    neither.

Every record a model cites also carries a currency word:

``current``
    every source the record binds by path still has the recorded digest.
``reviewed``
    every bound source that changed carries an accepted review in
    ``benchmarks/source_change_reviews.toml``: the source still has the reviewed
    bytes, and its code equals the code of the bytes the record measured, so
    only documentation or comments changed (:mod:`tools.source_change_reviews`).
    The reviewed paths are listed. The record was not measured again.
``outdated``
    at least one bound source changed or is gone without an accepted review;
    those paths are listed under ``changed``, and a review that exists but is
    rejected (a code change, other bytes, no measured bytes in the history) under
    ``review_rejected``. The record describes earlier bytes. It does not say the
    measured value changed.
``unbound``
    the record binds no source by path, so nothing can be said.

A record that keys its digests by lane names no path itself. Where the evidence
gate's manifest names the paths for it, those are used. Some records key their
digests by path under ``source_sha256`` or ``source_files_sha256`` instead of
``source_hashes``; where the manifest does not map such a record, those keys are
read.

A record under ``benchmarks/results`` in which no catalogue model's class name
appears measures no model in the sense above, yet it may bind sources. Every
such record that binds a source by path is listed with its currency too
(``uncited_bound_records``), and the totals over all records that parse are
given beside the totals of the cited ones. A review that applies to no changed
source of its record is listed under ``unused_source_change_reviews``, so it can
be removed once the record is measured again. An edit of any bound source
therefore changes this file.

In a repository it reads the files Git tracks and nothing else that lies in the
directories: a local run left beside the committed records, or a file Git
ignores, is not evidence and changes nothing. Outside a repository (a packaged
or temporary tree) it reads the directories as they are. It records no
timestamps, host names or commit hashes, and answers nothing about whether a
lane is built here today —
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
from collections.abc import Iterable, Mapping, Sequence
import json
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from sc_neurocore.neurons.models import _CLASS_TO_MODULE  # noqa: E402
from tools.benchmark_evidence_gate import (  # noqa: E402
    _SOURCE_PATH_SUFFIXES,
    GateFailure,
    _path_value,
    current_source_hash_failures,
    tracked_files,
)
from tools.source_change_reviews import Review, load_reviews, review_accepted  # noqa: E402

#: Schema of the emitted census.
CENSUS_SCHEMA = "sc-neurocore.benchmark-coverage-census.v4"

#: Where the census is committed.
DEFAULT_OUTPUT = "docs/_generated/benchmark_coverage_census.json"

#: Directory holding committed benchmark evidence.
RESULTS_DIR = "benchmarks/results"

#: Manifest of the evidence gate. It names the source paths of records that key
#: their digests by lane instead of by path.
GATE_MANIFEST = "benchmarks/benchmark_regression_gates.json"

#: Directory holding the benchmark scripts.
SCRIPTS_DIR = "benchmarks"

#: The three states a catalogue model can be in.
STATES = ("measured", "apparatus_only", "absent")

#: How a cited record is tied to the model it measures.
ATTRIBUTIONS = ("class_name", "bound_script")

#: The four things that can be said about a record's bound sources.
CURRENCIES = ("current", "reviewed", "outdated", "unbound")

#: Reasons of the evidence gate that mean a record binds no source by path.
_UNBOUND_REASONS = frozenset(
    {
        "artefact_is_missing_or_invalid",
        "missing_source_hashes",
        "missing_path_keyed_source_hashes",
    }
)

#: Fields besides ``source_hashes`` under which some records key their source
#: digests by repository path. They are read for a record that binds no path
#: under ``source_hashes`` and that the evidence gate's manifest does not map.
OTHER_PATH_KEYED_FIELDS = ("source_sha256", "source_files_sha256")


def model_names() -> tuple[str, ...]:
    """Return every registered catalogue model class name, sorted."""
    return tuple(sorted(_CLASS_TO_MODULE))


def _models_named_in(source: str) -> list[str]:
    """Return the catalogue models whose class name appears in *source* as a whole identifier.

    ``NMDANeuron`` is not named by a text that holds ``DendriticNMDANeuron``.
    """
    pattern = r"\b(?:" + "|".join(re.escape(name) for name in model_names()) + r")\b"
    return sorted(set(re.findall(pattern, source)))


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


def _evidence_files(repo_root: Path, directory: str, pattern: str) -> list[Path]:
    """Return the files below a directory that count: tracked ones in a repository, all otherwise.

    ``pattern`` is a glob relative to the directory (``**/*.json`` descends, ``*.py`` does not).
    """
    base = repo_root / directory
    tracked = tracked_files(repo_root, directory)
    if tracked is None:
        return sorted(base.glob(pattern))
    wanted = {repo_root / name for name in tracked}
    return sorted(path for path in base.glob(pattern) if path in wanted)


def _parsed_records(repo_root: Path) -> dict[str, object]:
    """Return every record under the results directory that parses, by its path below it."""
    results = repo_root / RESULTS_DIR
    parsed: dict[str, object] = {}
    for path in _evidence_files(repo_root, RESULTS_DIR, "**/*.json"):
        try:
            parsed[path.relative_to(results).as_posix()] = json.loads(
                path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            continue
    return parsed


def _bound_scripts(payload: object) -> list[str]:
    """Return the Python sources a record binds below the scripts directory, sorted.

    Each is given by its path below that directory. Only a script directly
    inside it is a benchmark script, and only those are looked up by the
    caller, so a bound source in a subfolder or elsewhere attributes nothing:
    it is a source, not the benchmark that wrote the record.
    """
    hashes = payload.get("source_hashes") if isinstance(payload, dict) else None
    if not isinstance(hashes, dict):
        return []
    prefix = f"{SCRIPTS_DIR}/"
    return sorted(
        key[len(prefix) :]
        for key in hashes
        if isinstance(key, str) and key.startswith(prefix) and key.endswith(".py")
    )


def record_attributions(repo_root: Path = REPO_ROOT) -> dict[str, dict[str, object]]:
    """Return, for each record that measures a model, the models and how that is known.

    A record that holds a model's class name measures that model
    (``class_name``). A record that holds none, and binds by path benchmark
    scripts that between them name exactly one model, measures that model
    (``bound_script``; the first such script by name is recorded). A record
    whose scripts name two models is credited to neither: which of them it
    measured is not stated anywhere a tool can read.

    Parameters
    ----------
    repo_root : Path
        Repository root; the results and script directories are resolved beneath it.

    Returns
    -------
    dict
        Record path below the results directory to ``models`` (sorted names),
        ``attribution`` (one of :data:`ATTRIBUTIONS`) and, for a script
        attribution, ``script``.
    """
    wanted = frozenset(model_names())
    named_by_script = {
        path.name: _models_named_in(source) for path, source in _script_sources(repo_root)
    }
    attributed: dict[str, dict[str, object]] = {}
    for record, payload in _parsed_records(repo_root).items():
        by_class = sorted(_names_in_payload(payload, wanted))
        if by_class:
            attributed[record] = {"models": by_class, "attribution": "class_name"}
            continue
        naming = [script for script in _bound_scripts(payload) if named_by_script.get(script)]
        models = {name for script in naming for name in named_by_script[script]}
        if len(models) == 1:
            attributed[record] = {
                "models": sorted(models),
                "attribution": "bound_script",
                "script": naming[0],
            }
    return attributed


def measured_models(repo_root: Path = REPO_ROOT) -> dict[str, list[str]]:
    """Return each model that a committed benchmark record measures, with the records.

    Parameters
    ----------
    repo_root : Path
        Repository root; the results directory is resolved beneath it.

    Returns
    -------
    dict
        Model name to the sorted records attributed to it (see
        :func:`record_attributions`), each given by its path below the results
        directory. A model absent from the mapping has no committed benchmark
        evidence at all.
    """
    found: dict[str, set[str]] = {}
    for record, entry in record_attributions(repo_root).items():
        models = entry["models"]
        assert isinstance(models, list)
        for name in models:
            found.setdefault(name, set()).add(record)
    return {name: sorted(records) for name, records in sorted(found.items())}


def manifest_source_paths(repo_root: Path = REPO_ROOT) -> dict[str, dict[str, str]]:
    """Return the source paths the evidence gate's manifest names, per record.

    Parameters
    ----------
    repo_root : Path
        Repository root; the manifest is read at :data:`GATE_MANIFEST` beneath it.

    Returns
    -------
    dict
        Record path below the results directory to its mapping of source path
        to the place of the digest inside the record. Empty when the manifest
        is absent or malformed.
    """
    try:
        manifest = json.loads((repo_root / GATE_MANIFEST).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    gates = manifest.get("gates") if isinstance(manifest, dict) else None
    prefix = f"{RESULTS_DIR}/"
    named: dict[str, dict[str, str]] = {}
    for gate in gates if isinstance(gates, list) else []:
        if not isinstance(gate, dict):
            continue
        artefact, mapping = gate.get("artefact"), gate.get("source_hashes")
        if isinstance(artefact, str) and artefact.startswith(prefix) and isinstance(mapping, dict):
            named[artefact.removeprefix(prefix)] = {str(k): str(v) for k, v in mapping.items()}
    return named


def _binds_no_path(failures: Sequence[GateFailure]) -> bool:
    """Say whether the gate's failures only state that no source is named by a path."""
    return bool(failures) and all(failure.reason in _UNBOUND_REASONS for failure in failures)


def _other_path_keyed_digests(artefact: Path) -> dict[str, str]:
    """Map each source a record keys by path under another digest field to its place.

    Parameters
    ----------
    artefact : Path
        The record file.

    Returns
    -------
    dict
        Source path to the dotted place of its digest inside the record, for
        every key under :data:`OTHER_PATH_KEYED_FIELDS` that the evidence gate
        would take for a repository path. Empty when the record cannot be read
        or names no such path.

    Notes
    -----
    A field is read only when at least one of its keys names a folder. A map
    of bare file names is keyed relative to a folder the record does not
    state; read from the repository root, every source would look removed.
    Beside keys with a folder, a bare key such as ``Cargo.lock`` names the
    file at the root.
    """
    try:
        payload = json.loads(artefact.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    places: dict[str, str] = {}
    for field in OTHER_PATH_KEYED_FIELDS:
        digests = payload.get(field)
        if not isinstance(digests, dict):
            continue
        paths = [
            path
            for path in digests
            if isinstance(path, str) and path.endswith(_SOURCE_PATH_SUFFIXES)
        ]
        if not any("/" in path for path in paths):
            continue
        for path in paths:
            places.setdefault(path, f"{field}.{path}")
    return places


def _recorded_digest(artefact: Path, place: str) -> str | None:
    """Return the digest a record holds at a dotted place, or ``None`` when it holds none there."""
    try:
        value = _path_value(json.loads(artefact.read_text(encoding="utf-8")), place)
    except (OSError, ValueError, KeyError):
        return None
    return value if isinstance(value, str) else None


def _apply_reviews(
    record: str,
    changed: list[str],
    artefact: Path,
    places: Mapping[str, str],
    repo_root: Path,
    reviews: Mapping[tuple[str, str], Review],
) -> dict[str, object]:
    """Return the currency entry of a record with changed sources, after its reviews are checked.

    Parameters
    ----------
    record : str
        The record, below the results directory.
    changed : list of str
        The bound sources that changed, sorted.
    artefact : Path
        The record file.
    places : mapping of str to str
        Source path to the place of its digest, where it is not ``source_hashes.<path>``.
    repo_root : Path
        The repository root.
    reviews : mapping
        Every review, keyed by ``(record, path)``.

    Returns
    -------
    dict
        ``reviewed`` when every changed source has an accepted review, else ``outdated`` with the sources left under
        ``changed``, and any accepted or rejected reviews listed.
    """
    reviewed: list[str] = []
    rejected: list[str] = []
    for path in changed:
        review = reviews.get((record, path))
        if review is None:
            continue
        digest = _recorded_digest(artefact, places.get(path, f"source_hashes.{path}"))
        if digest is not None and review_accepted(review, repo_root, digest):
            reviewed.append(path)
        else:
            rejected.append(path)
    if len(reviewed) == len(changed):
        return {"name": record, "currency": "reviewed", "reviewed": reviewed}
    entry: dict[str, object] = {
        "name": record,
        "currency": "outdated",
        "changed": [path for path in changed if path not in reviewed],
    }
    if reviewed:
        entry["reviewed"] = reviewed
    if rejected:
        entry["review_rejected"] = rejected
    return entry


def record_currency(
    record: str,
    repo_root: Path = REPO_ROOT,
    source_paths: Mapping[str, str] | None = None,
    reviews: Mapping[tuple[str, str], Review] | None = None,
) -> dict[str, object]:
    """Say whether the sources a record binds by path are still the same files.

    Parameters
    ----------
    record : str
        Path of the record below the results directory.
    repo_root : Path
        Repository root whose committed sources are compared.
    source_paths : mapping of str to str, optional
        Source path to the place of its digest inside the record, as the
        evidence gate's manifest names them. Used only when the record keys no
        digest by a path under ``source_hashes``; a named digest the record
        lacks counts as a changed source.
    reviews : mapping, optional
        Every source-change review, keyed by ``(record, path)``; read from the
        repository's reviews file when omitted.

    Returns
    -------
    dict
        ``name`` and ``currency`` (one of :data:`CURRENCIES`); an outdated
        record also lists the ``changed`` source paths, sorted, and a reviewed
        record the ``reviewed`` ones.

    Notes
    -----
    The digests are read from the first of three places that names a path:
    the record's ``source_hashes``, the manifest's map, and the record's own
    path keys under :data:`OTHER_PATH_KEYED_FIELDS`. The manifest comes before
    those fields because it is the evidence gate's declared map for records
    that key digests under another field, so both tools read the same sources.
    """
    artefact = repo_root / RESULTS_DIR / record
    failures = current_source_hash_failures(artefact, repo_root=repo_root)
    named: dict[str, str] = {}
    if source_paths and _binds_no_path(failures):
        failures = current_source_hash_failures(
            artefact, repo_root=repo_root, source_hash_paths=source_paths
        )
        named = {place: source for source, place in source_paths.items()}
    if _binds_no_path(failures):
        own = _other_path_keyed_digests(artefact)
        if own:
            failures = current_source_hash_failures(
                artefact, repo_root=repo_root, source_hash_paths=own
            )
            named = {place: source for source, place in own.items()}
    if not failures:
        return {"name": record, "currency": "current"}
    if all(failure.reason in _UNBOUND_REASONS for failure in failures):
        return {"name": record, "currency": "unbound"}
    changed = sorted(
        {
            named.get(failure.path, failure.path.split("->")[0].removeprefix("source_hashes."))
            for failure in failures
        }
    )
    return _apply_reviews(
        record,
        changed,
        artefact,
        {source: place for place, source in named.items()},
        repo_root,
        load_reviews(repo_root) if reviews is None else reviews,
    )


def _script_sources(repo_root: Path) -> list[tuple[Path, str]]:
    """Return every readable benchmark script with its text."""
    sources: list[tuple[Path, str]] = []
    for path in _evidence_files(repo_root, SCRIPTS_DIR, "*.py"):
        try:
            sources.append((path, path.read_text(encoding="utf-8")))
        except OSError:
            continue
    return sources


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
    found: dict[str, set[str]] = {}
    for path, source in _script_sources(repo_root):
        for name in _models_named_in(source):
            found.setdefault(name, set()).add(path.name)
    return {name: sorted(scripts) for name, scripts in sorted(found.items())}


def _currency_totals(entries: Iterable[dict[str, object]]) -> dict[str, int]:
    """Count the entries per currency word, every word present."""
    words = [entry["currency"] for entry in entries]
    return {word: words.count(word) for word in CURRENCIES}


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
        totals per state, totals of the cited records per currency word, the
        same totals over every record that parses, the records that bind
        sources although no model's name appears in them, and the reviews that
        apply to no changed source.
    """
    attributions = record_attributions(repo_root)
    measured: dict[str, list[str]] = {}
    for record, entry in sorted(attributions.items()):
        models = entry["models"]
        assert isinstance(models, list)
        for name in models:
            measured.setdefault(name, []).append(record)
    apparatus = apparatus_models(repo_root)
    named_paths = manifest_source_paths(repo_root)
    reviews = load_reviews(repo_root)
    every = {
        record: record_currency(record, repo_root, named_paths.get(record), reviews)
        for record in _parsed_records(repo_root)
    }
    applied: set[tuple[str, str]] = set()
    for record, entry in every.items():
        for key in ("changed", "reviewed", "review_rejected"):
            paths = entry.get(key, [])
            assert isinstance(paths, list)
            applied.update((record, str(path)) for path in paths)
    cited = set(attributions)
    currency = {
        record: {
            **every[record],
            **{key: value for key, value in attributions[record].items() if key != "models"},
        }
        for record in sorted(cited)
    }
    uncited_bound = [
        entry
        for record, entry in sorted(every.items())
        if record not in cited and entry["currency"] != "unbound"
    ]
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
                "records": [currency[record] for record in measured.get(name, [])],
                "scripts": apparatus.get(name, []),
            }
        )
    totals = {state: sum(1 for row in rows if row["state"] == state) for state in STATES}
    record_totals = _currency_totals(currency.values())
    return {
        "schema_version": CENSUS_SCHEMA,
        "catalogue_models": len(rows),
        "totals": totals,
        "cited_records": len(currency),
        "attribution_totals": {
            word: sum(1 for entry in currency.values() if entry["attribution"] == word)
            for word in ATTRIBUTIONS
        },
        "record_currency_totals": record_totals,
        "all_records": len(every),
        "all_record_currency_totals": _currency_totals(every.values()),
        "uncited_bound_records": uncited_bound,
        "unused_source_change_reviews": sorted(
            f"{record}:{path}" for record, path in reviews if (record, path) not in applied
        ),
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
    record_totals = census["record_currency_totals"]
    assert isinstance(record_totals, dict)
    by = census["attribution_totals"]
    assert isinstance(by, dict)
    lines.append(
        f"cited records: {census['cited_records']} "
        f"(by class name {by['class_name']}, through the bound script {by['bound_script']})"
    )
    for word in CURRENCIES:
        lines.append(f"  {word:15s} {int(record_totals[word]):4d}")
    all_totals = census["all_record_currency_totals"]
    assert isinstance(all_totals, dict)
    lines.append(f"all records: {census['all_records']}")
    for word in CURRENCIES:
        lines.append(f"  {word:15s} {int(all_totals[word]):4d}")
    others = census["uncited_bound_records"]
    assert isinstance(others, list)
    lines.append(f"bound to sources and named by no model: {len(others)}")
    unused = census["unused_source_change_reviews"]
    assert isinstance(unused, list)
    lines.append(f"source-change reviews that apply to nothing: {len(unused)}")
    lines.append("an outdated record describes earlier sources; it does not say the speed changed")
    lines.append(
        "a reviewed record's sources changed only in documentation or comments, each change reviewed"
    )
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

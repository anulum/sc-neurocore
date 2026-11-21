# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Split import gate for the planned SC-NC destinations

"""Report imports that cross the planned SC-NC destinations against their order.

SC-NeuroCore is to be split into SC-NC destinations whose imports form one
order: a destination may import only destinations earlier in it.
``tools/sc_nc_split_map.json`` names the destination of every top-level
package and module of ``src/sc_neurocore`` and the destinations each one may
import.

This gate reads the tracked Python files of ``src/sc_neurocore``, resolves
every absolute and relative import of ``sc_neurocore.<name>``, and of the
package root itself, which runs the facade, and reports:

- each import that the source's destination may not make, including an import
  into a facade or circuit-side name, neither of which belongs to the split graph;
- each import of SC-NeuroCore code from a circuit-side package, which must
  leave without it;
- every tracked package or module missing from the map, and every map entry
  without a tracked file.

Facade packages and modules stay in SC-NeuroCore and may import anything.
Only static ``import`` statements are seen: ``importlib`` calls and import
strings are not, and imports under ``if TYPE_CHECKING:`` are counted like any
other.

Usage::

    python tools/sc_nc_import_gate.py                         # report, exit 0
    python tools/sc_nc_import_gate.py --output report.json    # also write JSON
    python tools/sc_nc_import_gate.py --blocking SC-NC-CORE   # exit 1 on its findings

Exit status: 0 for a report or a clean blocking check, 1 when a destination
named with ``--blocking`` has findings, 2 when the map is malformed, when the
root tracks no Python file under ``src/sc_neurocore`` (an empty measurement is
refused rather than read as a clean one), or when a tracked file is missing or
does not decode or parse.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

DEFAULT_MAP = Path("tools/sc_nc_split_map.json")
SOURCE_ROOT = "src/sc_neurocore"
REPORT_SCHEMA = "sc-neurocore.split-import-gate.v1"


class MapError(ValueError):
    """Refuse a split map or a measurement that cannot be trusted."""


@dataclass(frozen=True)
class SplitMap:
    """Hold the destination order, import columns and the placement of every name.

    A name is a top-level package or module of ``src/sc_neurocore``; both are
    imported as ``sc_neurocore.<name>``, so they share one namespace.
    """

    order: tuple[str, ...]
    imports: dict[str, frozenset[str]]
    packages: dict[str, str]
    facade: frozenset[str]
    circuit_side: dict[str, str]


def load_map(path: Path) -> SplitMap:
    """Read and validate a split map.

    Parameters
    ----------
    path : Path
        The JSON split map.

    Returns
    -------
    SplitMap
        The validated map.

    Raises
    ------
    MapError
        If the file is not a valid map: an unknown destination, a column that
        names a destination not earlier in the order, or a package placed twice.
    """

    def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        keys = [key for key, _ in pairs]
        repeated = sorted({key for key in keys if keys.count(key) > 1})
        if repeated:
            raise ValueError(f"duplicate key(s) {repeated}")
        return dict(pairs)

    try:
        raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_keys)
    except (OSError, ValueError) as exc:
        raise MapError(f"{path}: cannot read the split map: {exc}") from exc
    order = tuple(raw.get("order", ()))
    if not order or len(set(order)) != len(order):
        raise MapError(f"{path}: 'order' must list each destination once")
    imports: dict[str, frozenset[str]] = {}
    for position, destination in enumerate(order):
        column = raw.get("imports", {}).get(destination)
        if column is None:
            raise MapError(f"{path}: destination {destination} has no import column")
        earlier = set(order[:position])
        later = sorted(set(column) - earlier)
        if later:
            raise MapError(
                f"{path}: {destination} may only import earlier destinations, not {later}"
            )
        imports[destination] = frozenset(column)
    unknown_columns = sorted(set(raw.get("imports", {})) - set(order))
    if unknown_columns:
        raise MapError(
            f"{path}: import columns for destinations outside the order: {unknown_columns}"
        )
    groups: list[list[str]] = [
        list(raw.get("packages", {})),
        list(raw.get("modules", {})),
        list(raw.get("facade_packages", ())),
        list(raw.get("facade_modules", ())),
        list(raw.get("circuit_side_packages", {})),
    ]
    seen: Counter[str] = Counter(name for group in groups for name in set(group))
    twice = sorted(name for name, count in seen.items() if count > 1)
    if twice:
        raise MapError(f"{path}: packages placed more than once: {twice}")
    placed: dict[str, str] = {**raw.get("packages", {}), **raw.get("modules", {})}
    unknown = sorted(set(placed.values()) - set(order))
    if unknown:
        raise MapError(f"{path}: packages placed in unknown destinations: {unknown}")
    facade = frozenset(groups[2]) | frozenset(groups[3])
    circuit_side: dict[str, str] = dict(raw.get("circuit_side_packages", {}))
    return SplitMap(order, imports, placed, facade, circuit_side)


def tracked_python_files(root: Path) -> list[str]:
    """Return the tracked Python files under ``src/sc_neurocore``.

    Parameters
    ----------
    root : Path
        The repository root.

    Returns
    -------
    list[str]
        Repository-relative POSIX paths, sorted.

    Raises
    ------
    MapError
        If the root is not a Git work tree, or tracks no Python file under
        ``src/sc_neurocore``: an empty measurement must not read as a clean one.
    """
    listed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--", SOURCE_ROOT],
        capture_output=True,
        check=False,
    )
    if listed.returncode != 0:
        raise MapError(f"{root}: cannot list tracked files: {listed.stderr.decode().strip()}")
    files = sorted(
        path for path in listed.stdout.decode("utf-8").split("\0") if path.endswith(".py")
    )
    if not files:
        raise MapError(f"{root}: no tracked Python file under {SOURCE_ROOT}")
    return files


def imported_packages(
    source: str, relative_path: str, names: frozenset[str]
) -> list[tuple[int, str]]:
    """Return the ``sc_neurocore`` top-level names a module imports.

    ``import sc_neurocore`` and ``from sc_neurocore import X``, where ``X`` is
    not a top-level package or module, import the package root itself; they
    are returned as ``__init__``, the root module that runs on that import.

    Parameters
    ----------
    source : str
        The module's source text.
    relative_path : str
        Its repository-relative path, used to resolve relative imports.
    names : frozenset[str]
        The top-level packages and modules the map places.

    Returns
    -------
    list[tuple[int, str]]
        ``(line, name)`` for each import of ``sc_neurocore`` or ``sc_neurocore.<name>``.

    Raises
    ------
    SyntaxError
        If the source does not parse.
    """
    # src/sc_neurocore/a/b.py is module sc_neurocore.a.b in package sc_neurocore.a;
    # an __init__.py is its package, so the containing package is the same slice.
    package_parts = PurePosixPath(relative_path).with_suffix("").parts[1:-1]
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source, filename=relative_path)):
        if isinstance(node, ast.Import):
            targets = [tuple(alias.name.split(".")) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package_parts[: len(package_parts) - (node.level - 1)]
                prefix = tuple(base) + (tuple(node.module.split(".")) if node.module else ())
            else:
                prefix = tuple(node.module.split(".")) if node.module else ()
            if prefix == ("sc_neurocore",):
                targets = [
                    prefix + ((alias.name,) if alias.name in names else ("__init__",))
                    for alias in node.names
                ]
            else:
                targets = [prefix]
        else:
            continue
        for parts in targets:
            if parts == ("sc_neurocore",):
                found.append((node.lineno, "__init__"))
            elif len(parts) >= 2 and parts[0] == "sc_neurocore":
                found.append((node.lineno, parts[1]))
    return found


def describe(split: SplitMap, name: str) -> str:
    """Name where a placed package or module goes in the split.

    Parameters
    ----------
    split : SplitMap
        The validated split map.
    name : str
        A placed top-level package or module.

    Returns
    -------
    str
        Its destination, ``SC-NeuroCore facade``, or ``circuit side (<repository>)``.
    """
    if name in split.facade:
        return "SC-NeuroCore facade"
    if name in split.circuit_side:
        return f"circuit side ({split.circuit_side[name]})"
    return split.packages[name]


def evaluate(root: Path, split: SplitMap, blocking: list[str]) -> dict[str, Any]:
    """Measure the tracked sources against the split map.

    A destination's import is a finding when its target lies outside the
    destination's column, or is a facade or circuit-side name. A circuit-side
    source's import is a finding when its target is SC-NeuroCore code: a
    destination or the facade. Facade sources are not restricted.

    Parameters
    ----------
    root : Path
        The repository root.
    split : SplitMap
        The validated split map.
    blocking : list[str]
        Destinations whose findings fail the gate.

    Returns
    -------
    dict[str, Any]
        The report: findings, unmapped and stale names, counts and status.
    """
    known = set(split.packages) | split.facade | set(split.circuit_side)
    findings: list[dict[str, Any]] = []
    unmapped: dict[str, set[str]] = {"packages": set(), "modules": set()}
    seen: set[str] = set()
    files = tracked_python_files(root)
    for relative_path in files:
        parts = PurePosixPath(relative_path).parts
        kind, name = (
            ("modules", PurePosixPath(parts[2]).stem) if len(parts) == 3 else ("packages", parts[2])
        )
        if name not in known:
            unmapped[kind].add(name)
            continue
        seen.add(name)
        if name in split.facade:
            continue
        source = describe(split, name)
        circuit_source = name in split.circuit_side
        try:
            text = (root / relative_path).read_text(encoding="utf-8")
            imported = imported_packages(text, relative_path, frozenset(known))
        except (OSError, UnicodeError, SyntaxError, ValueError) as exc:
            raise MapError(f"{relative_path}: cannot read or parse a tracked file: {exc}") from exc
        for line, target in imported:
            if target == name or target not in known:
                continue
            if circuit_source and target in split.circuit_side:
                continue
            target_destination = describe(split, target)
            if (
                not circuit_source
                and target in split.packages
                and (target_destination == source or target_destination in split.imports[source])
            ):
                continue
            findings.append(
                {
                    "file": relative_path,
                    "line": line,
                    "source": source,
                    "imports": f"sc_neurocore.{target}",
                    "target": target_destination,
                }
            )
    counts = Counter(finding["source"] for finding in findings)
    failed = sorted(destination for destination in blocking if counts[destination])
    return {
        "schema": REPORT_SCHEMA,
        "tracked_python_files": len(files),
        "findings_by_destination": {
            destination: counts[destination] for destination in split.order
        },
        "circuit_side_findings": sum(
            n for src, n in counts.items() if src.startswith("circuit side")
        ),
        "findings": findings,
        "unmapped_packages": sorted(unmapped["packages"]),
        "unmapped_modules": sorted(unmapped["modules"]),
        "stale_entries": sorted(known - seen),
        "blocking": sorted(blocking),
        "status": "fail" if failed else ("pass" if blocking else "report"),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the gate from the command line.

    Parameters
    ----------
    argv : list[str] | None
        Arguments; ``sys.argv[1:]`` when omitted.

    Returns
    -------
    int
        0 for a report or a clean blocking check, 1 for blocking findings, 2 for a
        malformed map or a root that tracks no Python file under ``src/sc_neurocore``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("."), help="repository root")
    parser.add_argument(
        "--map", type=Path, default=None, help="split map (default: tools/sc_nc_split_map.json)"
    )
    parser.add_argument("--output", type=Path, default=None, help="write the JSON report here")
    parser.add_argument(
        "--blocking", action="append", default=[], help="destination whose findings fail the gate"
    )
    args = parser.parse_args(argv)
    try:
        split = load_map(args.map or args.root / DEFAULT_MAP)
        unknown = sorted(set(args.blocking) - set(split.order))
        if unknown:
            raise MapError(f"--blocking names unknown destinations: {unknown}")
        report = evaluate(args.root, split, args.blocking)
    except MapError as exc:
        print(f"split import gate: {exc}", file=sys.stderr)
        return 2
    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for destination, count in report["findings_by_destination"].items():
        print(f"{destination}: {count} import(s) outside its column")
    print(f"circuit side: {report['circuit_side_findings']} import(s) of SC-NeuroCore code")
    for key, label in (
        ("unmapped_packages", "unmapped packages"),
        ("unmapped_modules", "unmapped modules"),
        ("stale_entries", "map entries without a tracked file"),
    ):
        if report[key]:
            print(f"{label}: {', '.join(report[key])}")
    print(f"status: {report['status']}")
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())

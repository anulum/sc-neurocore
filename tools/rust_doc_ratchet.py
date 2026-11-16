#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rust documentation debt may fall, never rise

"""Protect the engine library's original source cohort and individual doc debt.

Fresh Cargo JSON diagnostics identify missing documentation. The native syntax
parser resolves qualified declarations independently of warning color or line
positions. The lint is forced during this dedicated measurement so local lint
attributes and cap-lints cannot hide debt. Original immutable Git source is
compiled independently; candidate ceilings may only retain or reduce original
individual allowances. The default build configuration is measured, while
other targets and platform cfgs require qualification. An undocumented
macro-generated declaration is refused in the current source; in the original
baseline it counts as debt under an expansion identity.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Sequence
from pathlib import Path

from tools.doc_debt_ceiling import RatchetError, Verdict, read_ceiling
from tools.doc_debt_ceiling import compare as _compare
from tools.doc_debt_ceiling import write_ceiling as _write_ceiling
from tools.rust_doc_measurement import RustMeasurementError, measure_rust_findings
from tools.rust_doc_symbols import RustSymbolError, build_parser
from tools.rust_doc_history import INDIVIDUAL_SCHEMA, protect_rust_debt

__all__ = [
    "RUST_DOC_CEILING_SCHEMA_VERSION",
    "RatchetError",
    "Verdict",
    "ceiling_form",
    "compare",
    "main",
    "stable_argv",
    "measure",
    "read_ceiling",
    "write_ceiling",
]

#: Contract version of the ceiling record.
RUST_DOC_CEILING_SCHEMA_VERSION = "sc-neurocore.rust-doc-ceiling.v1"

REPO_ROOT = Path(__file__).resolve().parents[1]
#: Where the committed ceiling lives, beside the crate it constrains.
DEFAULT_CEILING = REPO_ROOT / "engine" / "missing_docs_ceiling.json"
#: What the record means, written for whoever opens the file before the tool.
CEILING_NOTE = (
    "Undocumented items in the engine crate, measured with rustc's own "
    "missing_docs lint. This ceiling may fall and may not rise: see "
    "tools/rust_doc_ratchet.py."
)


def compare(measured: int, ceiling: int) -> Verdict:
    """Return the verdict for one measurement against the ceiling."""
    return _compare(measured, ceiling, language="Rust")


def write_ceiling(path: Path, *, undocumented: int, files: int, provenance: dict[str, str]) -> None:
    """Write the crate's ceiling record, refusing to raise an existing one.

    Raises
    ------
    RatchetError
        The new figure is above the recorded one. Lowering is the tool's job;
        raising is a decision, and it belongs in a diff somebody signed.
    """
    _write_ceiling(
        path,
        undocumented=undocumented,
        files=files,
        note=CEILING_NOTE,
        schema_version=RUST_DOC_CEILING_SCHEMA_VERSION,
        provenance=provenance,
    )


def ceiling_form(path: Path) -> str:
    """Return the form ``--update`` keeps for a ceiling: its own schema, or the aggregate one for a new file.

    Parameters
    ----------
    path : Path
        The ceiling record.

    Returns
    -------
    str
        ``INDIVIDUAL_SCHEMA`` for a record that lists every case, else ``RUST_DOC_CEILING_SCHEMA_VERSION``.
    """
    if not path.is_file():
        return RUST_DOC_CEILING_SCHEMA_VERSION
    document = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(document, dict) and document.get("schema_version") == INDIVIDUAL_SCHEMA:
        return INDIVIDUAL_SCHEMA
    return RUST_DOC_CEILING_SCHEMA_VERSION


def stable_argv(argv: Sequence[str]) -> str:
    """Return the measurement's command line without its per-run ``--cfg sc_neurocore_doc_*`` arguments.

    The measurement passes rustc a digest of its inputs and a fresh execution id as cfgs. They belong in the full
    record; in the aggregate ceiling they would change the file on every update.

    Parameters
    ----------
    argv : sequence of str
        The measurement's argument list.

    Returns
    -------
    str
        The remaining arguments, joined by spaces.
    """
    kept: list[str] = []
    skip = False
    for index, argument in enumerate(argv):
        if skip:
            skip = False
            continue
        if (
            argument == "--cfg"
            and index + 1 < len(argv)
            and argv[index + 1].startswith("sc_neurocore_doc_")
        ):
            skip = True
            continue
        kept.append(argument)
    return " ".join(kept)


def _write_record(
    path: Path, *, undocumented: int, files: int, provenance: dict[str, object]
) -> None:
    """Write the full record of every case to ``path``, replacing any earlier record there.

    The record is written beside its target first and then moved into place, so a reader never sees half of it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".partial")
    staging.unlink(missing_ok=True)
    _write_ceiling(
        staging,
        undocumented=undocumented,
        files=files,
        note=CEILING_NOTE,
        schema_version=INDIVIDUAL_SCHEMA,
        provenance=provenance,
    )
    staging.replace(path)


def measure(root: Path, manifest: str) -> tuple[int, int, str]:
    """Run the lint and return the count, the file count and the rustc version.

    Raises
    ------
    RatchetError
        Cargo is unavailable, compilation fails or the compiler version cannot
        be obtained. No failed producer can supply a zero measurement or lower
        an existing ceiling.
    """
    if shutil.which("cargo") is None:
        raise RatchetError("cargo is not installed, so no measurement was taken")
    try:
        measured = measure_rust_findings(root, manifest)
    except RustMeasurementError as exc:
        raise RatchetError(str(exc)) from exc
    return measured.undocumented, measured.files, measured.rustc_version


def main(argv: list[str] | None = None) -> int:
    """Compare the crate against its ceiling, or lower the ceiling."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--manifest", default="engine/Cargo.toml")
    parser.add_argument("--ceiling", type=Path, default=DEFAULT_CEILING)
    parser.add_argument(
        "--update",
        action="store_true",
        help="lower the ceiling to the measured figure; it can never be raised. The ceiling keeps its form: an "
        "aggregate record stays aggregate, one that lists every case keeps listing them",
    )
    parser.add_argument(
        "--individual",
        action="store_true",
        help="with --update, write the ceiling in the individual form that lists every case, migrating an aggregate "
        "ceiling; a deliberate step, since an individual ceiling cannot return to the aggregate form",
    )
    parser.add_argument(
        "--record",
        type=Path,
        help="with --update, also write the full record of every undocumented case (the individual form) to "
        "RECORD, which must lie outside the repository",
    )
    args = parser.parse_args(argv)
    try:
        if args.record is not None and not args.update:
            raise RatchetError("--record needs --update")
        if args.individual and not args.update:
            raise RatchetError("--individual needs --update")
        root = args.repo.resolve()
        if args.record is not None and args.record.resolve().is_relative_to(root):
            raise RatchetError(
                f"the record must be written outside the repository, not at {args.record}"
            )
        if shutil.which("cargo") is None:
            raise RatchetError("cargo is not installed, so no measurement was taken")
        native = build_parser()
        measured = measure_rust_findings(root, args.manifest, parser=native)
        protected = protect_rust_debt(
            root, measured, args.ceiling, allow_create=args.update, parser=native
        )
        ceiling = (
            read_ceiling(args.ceiling) if args.ceiling.exists() else protected.original.undocumented
        )
        verdict = compare(measured.undocumented, ceiling)
        if args.update:
            if not verdict.ok:
                raise RatchetError("refusing to raise: " + verdict.summary())
            measured.verify(root)
            full: dict[str, object] = {
                "argv": list(measured.argv),
                "rustc": measured.rustc_version,
                "source_sha256": protected.revision,
                "manifest": measured.manifest,
                "debt_cases": [case.to_public_dict() for case in sorted(measured.cases)],
                "measurement": measured.to_public_dict(),
                "original_baseline": protected.to_public_dict(),
            }
            if args.individual or ceiling_form(args.ceiling) == INDIVIDUAL_SCHEMA:
                _write_ceiling(
                    args.ceiling,
                    undocumented=measured.undocumented,
                    files=measured.files,
                    note=CEILING_NOTE,
                    schema_version=INDIVIDUAL_SCHEMA,
                    provenance=full,
                )
            else:
                write_ceiling(
                    args.ceiling,
                    undocumented=measured.undocumented,
                    files=measured.files,
                    provenance={
                        "argv": stable_argv(measured.argv),
                        "rustc": measured.rustc_version,
                        "source_sha256": protected.revision,
                    },
                )
            if args.record is not None:
                _write_record(
                    args.record,
                    undocumented=measured.undocumented,
                    files=measured.files,
                    provenance=full,
                )
            print(
                f"Ceiling now {measured.undocumented} undocumented items in {measured.files} files."
            )
            return 0
        print(verdict.summary())
        return 0 if verdict.ok else 1
    except (RatchetError, RustMeasurementError, RustSymbolError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

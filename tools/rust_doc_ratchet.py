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
    "compare",
    "main",
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
        help="lower the ceiling to the measured figure; it can never be raised",
    )
    args = parser.parse_args(argv)
    try:
        if shutil.which("cargo") is None:
            raise RatchetError("cargo is not installed, so no measurement was taken")
        root = args.repo.resolve()
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
            _write_ceiling(
                args.ceiling,
                undocumented=measured.undocumented,
                files=measured.files,
                note=CEILING_NOTE,
                schema_version=INDIVIDUAL_SCHEMA,
                provenance={
                    "argv": list(measured.argv),
                    "rustc": measured.rustc_version,
                    "source_sha256": protected.revision,
                    "manifest": measured.manifest,
                    "debt_cases": [case.to_public_dict() for case in sorted(measured.cases)],
                    "measurement": measured.to_public_dict(),
                    "original_baseline": protected.to_public_dict(),
                },
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

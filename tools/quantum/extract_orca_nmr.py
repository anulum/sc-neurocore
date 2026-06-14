# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ORCA NMR parameter extraction for Posner evidence

"""Extract NMR shielding and J-coupling summaries from ORCA output.

The parser is intentionally conservative. It only writes a result for a
normally terminated ORCA output with a final single-point energy, a chemical
shielding summary, and an isotropic spin-spin coupling summary. The emitted JSON
is deterministic and includes source-file hashes for reproducible handoff into
SCPN-QUANTUM-CONTROL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0.0"

_NORMAL_TERMINATION_MARKER = "****ORCA TERMINATED NORMALLY****"
_FINAL_ENERGY_RE = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
_RUN_TIME_RE = re.compile(
    r"TOTAL RUN TIME:\s*(\d+)\s*days?\s*(\d+)\s*hours?\s*"
    r"(\d+)\s*minutes?\s*(\d+)\s*seconds?\s*(\d+)\s*msec"
)
_PROGRAM_VERSION_RE = re.compile(r"Program Version\s+([0-9][0-9.]*)")
_ROUTE_RE = re.compile(r"\|\s*\d+>\s*!\s*(.+)")
_CHARGE_RE = re.compile(r"Total Charge\s+Charge\s+\.+\s*(-?\d+)")
_MULT_RE = re.compile(r"Multiplicity\s+Mult\s+\.+\s*(\d+)")
_NEL_RE = re.compile(r"Number of Electrons\s+NEL\s+\.+\s*(\d+)")
_BASIS_DIM_RE = re.compile(r"Basis Dimension\s+Dim\s+\.+\s*(\d+)")
_HFTYP_RE = re.compile(r"Hartree-Fock type\s+HFTyp\s+\.+\s*(\S+)")
_SHIELDING_ROW_RE = re.compile(r"^\s*(\d+)\s+([A-Za-z]{1,2})\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$")
_PAIR_COUNT_RE = re.compile(r"Number of nuclear pairs to calculate something:\s*(\d+)")
_COUPLING_HEADER_RE = re.compile(r"^\s*((?:\d+\s+[A-Za-z]{1,2}\s*)+)$")
_COUPLING_ROW_RE = re.compile(r"^\s*(\d+)\s+([A-Za-z]{1,2})\s+(.+)$")


class OrcaNmrExtractionError(ValueError):
    """Raised when a required ORCA NMR section is missing or malformed."""


def parse_termination(text: str) -> dict[str, Any]:
    """Return normal termination and wall-clock duration details."""
    if _NORMAL_TERMINATION_MARKER not in text:
        raise OrcaNmrExtractionError("ORCA normal-termination marker not found.")
    match = _RUN_TIME_RE.search(text)
    if match is None:
        raise OrcaNmrExtractionError("TOTAL RUN TIME line not found in ORCA output.")
    days, hours, minutes, seconds, msec = (int(value) for value in match.groups())
    total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds + msec / 1000.0
    return {
        "normal_termination": True,
        "total_run_time_text": match.group(0).split(":", 1)[1].strip(),
        "total_run_time_seconds": total_seconds,
    }


def parse_final_energy(text: str) -> float:
    """Return the last ``FINAL SINGLE POINT ENERGY`` value in hartree."""
    matches = _FINAL_ENERGY_RE.findall(text)
    if not matches:
        raise OrcaNmrExtractionError("FINAL SINGLE POINT ENERGY not found in ORCA output.")
    return float(matches[-1])


def parse_run_settings(text: str) -> dict[str, Any]:
    """Parse basic ORCA route and SCF settings echoed in the output."""
    route = _ROUTE_RE.search(text)
    return {
        "program_version": _first_group(_PROGRAM_VERSION_RE, text),
        "route_line": route.group(1).strip() if route else None,
        "hartree_fock_type": _first_group(_HFTYP_RE, text),
        "charge": _int_or_none(_CHARGE_RE, text),
        "multiplicity": _int_or_none(_MULT_RE, text),
        "number_of_electrons": _int_or_none(_NEL_RE, text),
        "basis_dimension": _int_or_none(_BASIS_DIM_RE, text),
    }


def parse_shielding_summary(text: str) -> list[dict[str, Any]]:
    """Parse ORCA's ``CHEMICAL SHIELDING SUMMARY`` table."""
    marker = "CHEMICAL SHIELDING SUMMARY (ppm)"
    start = text.find(marker)
    if start == -1:
        raise OrcaNmrExtractionError("CHEMICAL SHIELDING SUMMARY section not found.")
    block = text[start:]
    rows: list[dict[str, Any]] = []
    for line in block.splitlines():
        match = _SHIELDING_ROW_RE.match(line)
        if match:
            rows.append(
                {
                    "atom_index": int(match.group(1)),
                    "element": match.group(2),
                    "isotropic_ppm": float(match.group(3)),
                    "anisotropy_ppm": float(match.group(4)),
                }
            )
        elif rows and not line.strip():
            break
    if not rows:
        raise OrcaNmrExtractionError("No shielding rows found in ORCA output.")
    return rows


def parse_isotropic_couplings(text: str) -> dict[str, Any]:
    """Parse the isotropic J-coupling matrix summary in hertz."""
    marker = "SUMMARY OF ISOTROPIC COUPLING CONSTANTS J (Hz)"
    start = text.find(marker)
    if start == -1:
        raise OrcaNmrExtractionError("Isotropic J-coupling summary section not found.")
    pair_count = _int_or_none(_PAIR_COUNT_RE, text)
    lines = text[start:].splitlines()
    headers: list[dict[str, Any]] | None = None
    rows: list[dict[str, Any]] = []
    matrix: dict[str, dict[str, float]] = {}
    nonzero_pairs: list[dict[str, Any]] = []

    for line in lines:
        if headers is None:
            header_match = _COUPLING_HEADER_RE.match(line)
            if header_match:
                tokens = header_match.group(1).split()
                headers = [
                    {"atom_index": int(tokens[index]), "element": tokens[index + 1]}
                    for index in range(0, len(tokens), 2)
                ]
            continue
        row_match = _COUPLING_ROW_RE.match(line)
        if row_match is None:
            if rows:
                break
            continue
        values = [float(value) for value in row_match.group(3).split()]
        if len(values) != len(headers):
            break
        row_label = f"{row_match.group(1)}{row_match.group(2)}"
        row = {
            "atom_index": int(row_match.group(1)),
            "element": row_match.group(2),
            "values_hz": values,
        }
        rows.append(row)
        matrix[row_label] = {
            f"{header['atom_index']}{header['element']}": values[index]
            for index, header in enumerate(headers)
        }

    if headers is None or not rows:
        raise OrcaNmrExtractionError("No isotropic J-coupling matrix rows found.")

    for row_index, row in enumerate(rows):
        for col_index in range(row_index + 1, len(headers)):
            value = row["values_hz"][col_index]
            if value != 0.0:
                col = headers[col_index]
                nonzero_pairs.append(
                    {
                        "atom_a": row["atom_index"],
                        "element_a": row["element"],
                        "atom_b": col["atom_index"],
                        "element_b": col["element"],
                        "j_iso_hz": value,
                    }
                )

    return {
        "reported_pair_count": pair_count,
        "matrix_labels": [f"{item['atom_index']}{item['element']}" for item in headers],
        "matrix_hz": matrix,
        "nonzero_pairs": nonzero_pairs,
    }


def extract_orca_nmr(
    input_path: Path,
    *,
    extra_sources: list[Path] | None = None,
    required_shielding_elements: tuple[str, ...] = ("P",),
    require_couplings: bool = True,
) -> dict[str, Any]:
    """Parse a completed ORCA NMR output into a deterministic parameter record."""
    if not input_path.is_file():
        raise OrcaNmrExtractionError(f"ORCA output file not found: {input_path}")
    text = input_path.read_text(encoding="utf-8", errors="replace")
    shielding = parse_shielding_summary(text)
    couplings = parse_isotropic_couplings(text) if require_couplings else None

    present = {entry["element"] for entry in shielding}
    missing = [element for element in required_shielding_elements if element not in present]
    if missing:
        raise OrcaNmrExtractionError(
            "Required shielding elements missing from ORCA output: " + ", ".join(missing)
        )

    sources = [_source_record(input_path, role="orca_output")]
    for extra in extra_sources or []:
        if not extra.is_file():
            raise OrcaNmrExtractionError(f"Provenance source file not found: {extra}")
        sources.append(_source_record(extra, role="provenance"))

    by_element: dict[str, list[dict[str, Any]]] = {}
    for entry in shielding:
        by_element.setdefault(entry["element"], []).append(entry)

    return {
        "schema_version": SCHEMA_VERSION,
        "final_single_point_energy_eh": parse_final_energy(text),
        "termination": parse_termination(text),
        "run_settings": parse_run_settings(text),
        "chemical_shielding": {
            "nucleus_count": len(shielding),
            "by_element": {element: by_element[element] for element in sorted(by_element)},
        },
        "spin_spin_coupling": couplings,
        "provenance": {"sources": sources},
    }


def _source_record(path: Path, *, role: str) -> dict[str, str]:
    resolved = path.resolve()
    return {
        "role": role,
        "path": str(resolved),
        "name": resolved.name,
        "sha256": _sha256_file(resolved),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_group(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(1) if match else None


def _int_or_none(pattern: re.Pattern[str], text: str) -> int | None:
    match = pattern.search(text)
    return int(match.group(1)) if match else None


def serialise(payload: dict[str, Any]) -> str:
    """Serialise a parameter record to deterministic JSON with a trailing newline."""
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract ORCA NMR shielding and J-coupling summaries into JSON.",
    )
    parser.add_argument("--input", required=True, type=Path, help="Completed ORCA NMR .out file.")
    parser.add_argument("--output", type=Path, default=None, help="JSON output path.")
    parser.add_argument(
        "--source",
        dest="extra_sources",
        action="append",
        type=Path,
        default=[],
        help="Additional provenance file to hash (repeatable).",
    )
    parser.add_argument(
        "--require-shielding-element",
        dest="required_shielding_elements",
        action="append",
        default=None,
        help="Element symbol whose shielding rows must be present (default: P).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    required = (
        tuple(args.required_shielding_elements) if args.required_shielding_elements else ("P",)
    )
    try:
        payload = extract_orca_nmr(
            args.input,
            extra_sources=list(args.extra_sources),
            required_shielding_elements=required,
        )
    except OrcaNmrExtractionError as error:
        print(f"extract_orca_nmr: {error}", file=sys.stderr)
        return 1
    rendered = serialise(payload)
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

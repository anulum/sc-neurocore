# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ORCA EPR/HFC parameter extraction for Posner evidence

r"""Extract EPR/HFC parameters from an ORCA single-point output into JSON.

This tool reads a completed ORCA ``EPRNMR`` output file (gtensor + hyperfine
property run) and produces a deterministic JSON record carrying:

- the final single-point SCF energy and the property-module reference energy,
- the normal-termination marker and the total wall-clock run time,
- the route/method line, charge, multiplicity, electron count, and basis
  dimension echoed by ORCA,
- the electronic g-matrix (raw 3x3, principal values, isotropic value,
  Delta-g principal values and isotropic value),
- the per-nucleus hyperfine structure for every requested nucleus
  (A(FC), A(SD), A(ORB) where present, A(Tot) principal values, A(iso)),
- SHA-256 hashes and absolute paths of the source output and any additional
  provenance files supplied on the command line.

Every value is parsed from the ORCA output; no constant is hand-copied. The
extractor fails closed: if any required section (normal termination, final
energy, g-matrix, at least one phosphorus and one calcium hyperfine entry) is
missing, it raises :class:`OrcaExtractionError` and writes no output file.

The JSON is emitted with sorted keys and two-space indentation so that running
the tool twice against the same inputs produces byte-identical output.
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

_PROGRAM_VERSION_RE = re.compile(r"Program Version\s+([0-9][0-9.]*)")
_FINAL_ENERGY_RE = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
_PROPERTY_ENERGY_RE = re.compile(r"^Energy\s+:\s+(-?\d+\.\d+)\s+Eh", re.MULTILINE)
_RUN_TIME_RE = re.compile(
    r"TOTAL RUN TIME:\s*(\d+)\s*days?\s*(\d+)\s*hours?\s*"
    r"(\d+)\s*minutes?\s*(\d+)\s*seconds?\s*(\d+)\s*msec"
)
_ROUTE_RE = re.compile(r"\|\s*\d+>\s*!\s*(.+)")
_CHARGE_RE = re.compile(r"Total Charge\s+Charge\s+\.+\s*(-?\d+)")
_MULT_RE = re.compile(r"Multiplicity\s+Mult\s+\.+\s*(\d+)")
_NEL_RE = re.compile(r"Number of Electrons\s+NEL\s+\.+\s*(\d+)")
_BASIS_DIM_RE = re.compile(r"Basis Dimension\s+Dim\s+\.+\s*(\d+)")
_HFTYP_RE = re.compile(r"Hartree-Fock type\s+HFTyp\s+\.+\s*(\S+)")

_G_MATRIX_HEADER = "ELECTRONIC G-MATRIX"
_G_MATRIX_RAW_RE = re.compile(
    r"The g-matrix:\s*\n"
    r"\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*\n"
    r"\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*\n"
    r"\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*\n"
)
_G_TOT_RE = re.compile(
    r"g\(tot\)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+iso=\s*(-?\d+\.\d+)"
)
_DELTA_G_RE = re.compile(
    r"Delta-g\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+iso=\s*(-?\d+\.\d+)"
)

_HFC_SECTION_HEADER = "ELECTRIC AND MAGNETIC HYPERFINE STRUCTURE"
_NUCLEUS_HEADER_RE = re.compile(
    r"Nucleus\s+(\d+)([A-Za-z]{1,2})\s*:\s*A\s*:\s*"
    r"Isotope=\s*(\d+)\s+I=\s*([\d.]+)\s+P=\s*(-?[\d.]+)\s*MHz/au\*\*3"
)
_A_FC_RE = re.compile(r"A\(FC\)\s+(-?\d+\.\d+)")
_A_SD_RE = re.compile(r"A\(SD\)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")
_A_ORB_RE = re.compile(
    r"A\(ORB\)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+A\(PC\)\s*=\s*(-?\d+\.\d+)"
)
_A_TOT_RE = re.compile(
    r"A\(Tot\)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+A\(iso\)=\s*(-?\d+\.\d+)"
)


class OrcaExtractionError(ValueError):
    """Raised when a required ORCA section is missing or malformed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_termination(text: str) -> dict[str, Any]:
    """Parse the normal-termination marker and total run time.

    Raises :class:`OrcaExtractionError` when the normal-termination marker or
    the total-run-time line is absent — an aborted run carries no trustworthy
    property values.
    """
    normal = _NORMAL_TERMINATION_MARKER in text
    if not normal:
        raise OrcaExtractionError(
            "ORCA normal-termination marker not found; refusing to extract "
            "parameters from a non-converged or aborted run."
        )
    match = _RUN_TIME_RE.search(text)
    if match is None:
        raise OrcaExtractionError("TOTAL RUN TIME line not found in ORCA output.")
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
        raise OrcaExtractionError("FINAL SINGLE POINT ENERGY not found in ORCA output.")
    return float(matches[-1])


def parse_run_settings(text: str) -> dict[str, Any]:
    """Parse the route line and the echoed general-settings block."""
    route_match = _ROUTE_RE.search(text)
    charge_match = _CHARGE_RE.search(text)
    mult_match = _MULT_RE.search(text)
    return {
        "program_version": _first_group(_PROGRAM_VERSION_RE, text),
        "route_line": route_match.group(1).strip() if route_match else None,
        "hartree_fock_type": _first_group(_HFTYP_RE, text),
        "charge": int(charge_match.group(1)) if charge_match else None,
        "multiplicity": int(mult_match.group(1)) if mult_match else None,
        "number_of_electrons": _int_or_none(_NEL_RE, text),
        "basis_dimension": _int_or_none(_BASIS_DIM_RE, text),
        "property_module_energy_eh": _last_property_energy(text),
    }


def parse_g_matrix(text: str) -> dict[str, Any]:
    """Parse the electronic g-matrix, principal g-values and Delta-g values."""
    start = text.find(_G_MATRIX_HEADER)
    if start == -1:
        raise OrcaExtractionError("ELECTRONIC G-MATRIX section not found in ORCA output.")
    block = text[start:]
    raw = _G_MATRIX_RAW_RE.search(block)
    g_tot = _G_TOT_RE.search(block)
    delta = _DELTA_G_RE.search(block)
    if raw is None or g_tot is None or delta is None:
        raise OrcaExtractionError("ELECTRONIC G-MATRIX block is incomplete or malformed.")
    raw_values = [float(value) for value in raw.groups()]
    return {
        "g_matrix": [raw_values[0:3], raw_values[3:6], raw_values[6:9]],
        "g_principal": [float(value) for value in g_tot.groups()[:3]],
        "g_isotropic": float(g_tot.group(4)),
        "delta_g_principal": [float(value) for value in delta.groups()[:3]],
        "delta_g_isotropic": float(delta.group(4)),
    }


def _parse_nucleus_block(header: re.Match[str], block: str) -> dict[str, Any]:
    a_tot = _A_TOT_RE.search(block)
    a_fc = _A_FC_RE.search(block)
    a_sd = _A_SD_RE.search(block)
    if a_tot is None or a_fc is None or a_sd is None:
        raise OrcaExtractionError(
            f"Hyperfine block for nucleus {header.group(1)}{header.group(2)} "
            "is missing A(FC), A(SD) or A(Tot)."
        )
    a_orb = _A_ORB_RE.search(block)
    return {
        "atom_index": int(header.group(1)),
        "element": header.group(2),
        "isotope": int(header.group(3)),
        "spin_quantum_number": float(header.group(4)),
        "prefactor_mhz_per_au3": float(header.group(5)),
        "a_fc_isotropic_mhz": float(a_fc.group(1)),
        "a_sd_principal_mhz": [float(value) for value in a_sd.groups()],
        "a_orb_principal_mhz": ([float(value) for value in a_orb.groups()[:3]] if a_orb else None),
        "a_orb_isotropic_mhz": float(a_orb.group(4)) if a_orb else None,
        "a_tot_principal_mhz": [float(value) for value in a_tot.groups()[:3]],
        "a_isotropic_mhz": float(a_tot.group(4)),
    }


def parse_hyperfine(text: str) -> list[dict[str, Any]]:
    """Parse every per-nucleus hyperfine block in the EPR/NMR section."""
    section_start = text.find(_HFC_SECTION_HEADER)
    if section_start == -1:
        raise OrcaExtractionError("ELECTRIC AND MAGNETIC HYPERFINE STRUCTURE section not found.")
    section = text[section_start:]
    headers = list(_NUCLEUS_HEADER_RE.finditer(section))
    if not headers:
        raise OrcaExtractionError("No hyperfine nucleus blocks found in ORCA output.")
    nuclei: list[dict[str, Any]] = []
    for index, header in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(section)
        nuclei.append(_parse_nucleus_block(header, section[header.start() : end]))
    return nuclei


def _group_hyperfine_by_element(nuclei: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for nucleus in nuclei:
        grouped.setdefault(nucleus["element"], []).append(nucleus)
    return grouped


def extract_orca_parameters(
    input_path: Path,
    *,
    extra_sources: list[Path] | None = None,
    required_elements: tuple[str, ...] = ("P", "Ca"),
) -> dict[str, Any]:
    """Parse an ORCA EPR/HFC output into a deterministic parameter record.

    Args:
        input_path: ORCA ``.out`` file from a completed gtensor/HFC run.
        extra_sources: additional provenance files to hash (input geometry,
            ``.inp``, reproducibility log) recorded alongside the output.
        required_elements: element symbols whose hyperfine entries must be
            present; extraction fails closed when any is absent.

    Returns:
        A nested dict ready for deterministic JSON serialisation.

    Raises:
        OrcaExtractionError: when any required section or element is missing.
    """
    if not input_path.is_file():
        raise OrcaExtractionError(f"ORCA output file not found: {input_path}")
    text = input_path.read_text(encoding="utf-8", errors="replace")

    termination = parse_termination(text)
    final_energy = parse_final_energy(text)
    settings = parse_run_settings(text)
    g_matrix = parse_g_matrix(text)
    nuclei = parse_hyperfine(text)
    grouped = _group_hyperfine_by_element(nuclei)

    missing = [element for element in required_elements if not grouped.get(element)]
    if missing:
        raise OrcaExtractionError(
            "Required hyperfine elements missing from ORCA output: " + ", ".join(missing)
        )

    sources = [_source_record(input_path, role="orca_output")]
    for extra in extra_sources or []:
        if not extra.is_file():
            raise OrcaExtractionError(f"Provenance source file not found: {extra}")
        sources.append(_source_record(extra, role="provenance"))

    return {
        "schema_version": SCHEMA_VERSION,
        "final_single_point_energy_eh": final_energy,
        "termination": termination,
        "run_settings": settings,
        "g_tensor": g_matrix,
        "hyperfine": {
            "nucleus_count": len(nuclei),
            "by_element": {element: grouped[element] for element in sorted(grouped)},
        },
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


def _first_group(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(1) if match else None


def _int_or_none(pattern: re.Pattern[str], text: str) -> int | None:
    match = pattern.search(text)
    return int(match.group(1)) if match else None


def _last_property_energy(text: str) -> float | None:
    matches = _PROPERTY_ENERGY_RE.findall(text)
    return float(matches[-1]) if matches else None


def serialise(payload: dict[str, Any]) -> str:
    """Serialise a parameter record to deterministic JSON with a trailing newline."""
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract ORCA EPR/HFC parameters into deterministic JSON.",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the completed ORCA EPR/HFC .out file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write the JSON record (default: stdout).",
    )
    parser.add_argument(
        "--source",
        dest="extra_sources",
        action="append",
        type=Path,
        default=[],
        help="Additional provenance file to hash (repeatable).",
    )
    parser.add_argument(
        "--require-element",
        dest="required_elements",
        action="append",
        default=None,
        help="Element symbol whose hyperfine entry must be present "
        "(repeatable; default: P and Ca).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    required = tuple(args.required_elements) if args.required_elements else ("P", "Ca")
    try:
        payload = extract_orca_parameters(
            args.input,
            extra_sources=list(args.extra_sources),
            required_elements=required,
        )
    except OrcaExtractionError as error:
        print(f"extract_orca_params: {error}", file=sys.stderr)
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

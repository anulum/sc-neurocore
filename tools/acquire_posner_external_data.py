# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Posner external data acquisition workflow

"""Acquire external molecular and backend data for Posner verification.

This tool prepares reproducible quantum-chemistry jobs, parses completed
ORCA HFC outputs into verification JSON files, and retrieves IBM backend
calibration snapshots. It deliberately refuses to create complete runtime
parameter files from guesses.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

from orca_posner_hf import (  # noqa: E402
    auto_a_ref,
    compute_qubit_dipolar_tensor_table_from_xyz,
    convert_to_dimensionless,
    get_all_coords_flat,
    group_by_site,
    parse_orca_hf_output,
    read_xyz_phosphorus_coords,
)

_SPIN_KEYS = ("Axx", "Ayy", "Azz", "Axy", "Axz", "Ayz")
_DEFAULT_OUT = _ROOT / "results" / "posner_external_data"
_CA_START_BY_LABEL = {
    "Ca1": 8,
    "Ca2": 11,
    "Ca3": 14,
    "Ca4": 17,
    "Ca5": 20,
    "Ca6": 23,
    "Ca7": 26,
    "Ca8": 29,
    "Ca_c": 32,
}
_POSNER_ELECTRONS_NEUTRAL = 9 * 20 + 6 * 15 + 24 * 8


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _jsonable(obj: Any) -> Any:
    """Convert Qiskit runtime objects to JSON-compatible data."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        return _jsonable(to_dict())
    return str(obj)


def _call_optional(fn: Any, *args: Any, **kwargs: Any) -> Any:
    if not callable(fn):
        return None
    try:
        return fn(*args, **kwargs)
    except TypeError:
        return fn()


def _normalise_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", label.lower())


def _vault_section(path: Path, section_regex: str) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(
        rf"(?ims)^#{{1,6}}\s*{section_regex}[^\n]*\n(?P<body>.*?)(?=^#{{1,6}}\s|\Z)"
    )
    match = pattern.search(text)
    return match.group("body") if match else ""


def _vault_field(path: Path | None, section_regex: str, labels: tuple[str, ...]) -> str | None:
    """Read a labelled Vault field without printing or persisting secrets."""
    if path is None or not path.exists():
        return None
    body = _vault_section(path, section_regex)
    wanted = {_normalise_label(label) for label in labels}
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        label, value = line.split(":", 1)
        if _normalise_label(label.strip("-* `")) not in wanted:
            continue
        backticked = re.findall(r"`([^`]+)`", value)
        value = backticked[0] if backticked else value.strip()
        value = value.strip().strip("`").strip()
        if not value or value.lower() in {"none", "null", "disabled"}:
            return None
        return value
    return None


def _orca_qc_path() -> str | None:
    """Return a likely ORCA quantum-chemistry executable, excluding GNOME Orca."""
    env = os.environ.get("ORCA_QC_BIN")
    candidates = [env] if env else []
    found = shutil.which("orca")
    if found:
        candidates.append(found)
    for candidate in candidates:
        if not candidate:
            continue
        try:
            proc = subprocess.run(
                [candidate, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        combined = f"{proc.stdout}\n{proc.stderr}".lower()
        if "screen reader" in combined or "gnome" in combined:
            continue
        if "orca" in combined or re.search(r"\b[4-9]\.\d", combined):
            return candidate
    return None


def _find_orca_qc_binary(root: Path) -> Path | None:
    candidates = [root / "orca", *root.glob("**/orca")]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            proc = subprocess.run(
                [str(candidate), "--version"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        combined = f"{proc.stdout}\n{proc.stderr}".lower()
        if "screen reader" in combined or "gnome" in combined:
            continue
        if "orca" in combined or re.search(r"\b[4-9]\.\d", combined):
            return candidate
    return None


def install_orca(args: argparse.Namespace) -> int:
    """Install a user-provided official ORCA QC archive/installer."""
    archive = Path(args.archive).expanduser().resolve()
    if not archive.exists():
        raise SystemExit(
            f"ORCA archive not found: {archive}. Download the official Linux "
            "installer/archive from the ORCA forum after accepting the EULA."
        )
    prefix = Path(args.prefix).expanduser().resolve()
    prefix.mkdir(parents=True, exist_ok=True)

    name = archive.name.lower()
    if name.endswith(".run"):
        archive.chmod(archive.stat().st_mode | 0o700)
        subprocess.run([str(archive), "--", "-p", str(prefix)], check=True)
    elif any(name.endswith(suffix) for suffix in (".tar.xz", ".tar.gz", ".tgz", ".zip")):
        shutil.unpack_archive(str(archive), str(prefix))
    else:
        raise SystemExit(
            "Unsupported ORCA installer format. Expected .run, .tar.xz, .tar.gz, .tgz, or .zip"
        )

    qc_bin = _find_orca_qc_binary(prefix)
    if qc_bin is None:
        raise SystemExit(
            f"ORCA QC executable was not found after extracting/installing into {prefix}"
        )
    if args.env_file:
        env_file = Path(args.env_file)
        env_file.parent.mkdir(parents=True, exist_ok=True)
        env_file.write_text(
            f"export ORCA_QC_BIN={qc_bin}\nexport PATH={qc_bin.parent}:$PATH\n",
            encoding="utf-8",
        )
    print(f"Installed ORCA QC executable: {qc_bin}")
    if args.env_file:
        print(f"Wrote environment snippet: {args.env_file}")
    return 0


def _orca_header(
    *,
    functional: str,
    basis: str,
    scf: str,
    n_cores: int,
    maxcore_mb: int,
    extra: str,
) -> list[str]:
    lines = [
        f"! {functional} {basis} D3BJ RIJCOSX {scf} DefGrid3 {extra}".strip(),
        f"%maxcore {maxcore_mb}",
    ]
    if n_cores > 1:
        lines.extend(["%pal", f"  nprocs {n_cores}", "end"])
    return lines


def _validate_charge_multiplicity(charge: int, multiplicity: int) -> None:
    if multiplicity < 1:
        raise ValueError(f"ORCA multiplicity must be >= 1, got {multiplicity}")
    electrons = _POSNER_ELECTRONS_NEUTRAL - charge
    unpaired = multiplicity - 1
    if electrons < 1 or (electrons - unpaired) % 2 != 0:
        raise ValueError(
            "Invalid Posner charge/multiplicity: "
            f"charge={charge}, multiplicity={multiplicity} gives {electrons} "
            "electrons, which is incompatible with the requested spin state"
        )


def _xyz_block(charge: int, multiplicity: int) -> list[str]:
    _validate_charge_multiplicity(charge, multiplicity)
    lines = [f"* xyz {charge} {multiplicity}"]
    for elem, coord in get_all_coords_flat():
        lines.append(f"  {elem:2s} {coord[0]:14.8f} {coord[1]:14.8f} {coord[2]:14.8f}")
    lines.append("*")
    return lines


def _xyzfile_block(charge: int, multiplicity: int, xyz_name: str) -> list[str]:
    _validate_charge_multiplicity(charge, multiplicity)
    return [f"* xyzfile {charge} {multiplicity} {xyz_name}"]


def _eprnmr_block() -> list[str]:
    return [
        "%eprnmr",
        "  gtensor = true",
        "  nuclei = all P {aiso, adip, aorb}",
        "  nuclei = all Ca {aiso, adip}",
        "  printlevel = 5",
        "end",
    ]


def _input_text(
    *,
    charge: int,
    multiplicity: int,
    functional: str,
    basis: str,
    scf: str,
    n_cores: int,
    maxcore_mb: int,
    job: str,
    xyzfile: str | None = None,
    epr: bool = False,
) -> str:
    lines = _orca_header(
        functional=functional,
        basis=basis,
        scf=scf,
        n_cores=n_cores,
        maxcore_mb=maxcore_mb,
        extra=job,
    )
    if epr:
        lines.extend(["", *_eprnmr_block()])
    lines.append("")
    if xyzfile:
        lines.extend(_xyzfile_block(charge, multiplicity, xyzfile))
    else:
        lines.extend(_xyz_block(charge, multiplicity))
    lines.append("")
    return "\n".join(lines)


def prepare_orca(args: argparse.Namespace) -> int:
    """Write ORCA input deck templates and provenance manifest."""
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    qc_orca = _orca_qc_path()

    _write(
        out / "00_posner_neutral_opt.inp",
        _input_text(
            charge=0,
            multiplicity=1,
            functional=args.functional,
            basis=args.basis,
            scf=args.scf,
            n_cores=args.n_cores,
            maxcore_mb=args.maxcore_mb,
            job="Opt Freq",
        ),
    )
    _write(
        out / "01_posner_cation_radical_epr.inp.template",
        _input_text(
            charge=1,
            multiplicity=2,
            functional=f"U{args.functional}"
            if not args.functional.startswith("U")
            else args.functional,
            basis=args.basis,
            scf=args.scf,
            n_cores=args.n_cores,
            maxcore_mb=args.maxcore_mb,
            job="SP",
            xyzfile="00_posner_neutral_opt.xyz",
            epr=True,
        ),
    )
    _write(
        out / "01_posner_cation_radical_relaxed_opt.inp",
        _input_text(
            charge=1,
            multiplicity=2,
            functional=f"U{args.functional}"
            if not args.functional.startswith("U")
            else args.functional,
            basis=args.basis,
            scf=args.scf,
            n_cores=args.n_cores,
            maxcore_mb=args.maxcore_mb,
            job="Opt Freq",
            xyzfile="00_posner_neutral_opt.xyz",
        ),
    )
    _write(
        out / "02_posner_cation_radical_relaxed_epr.inp.template",
        _input_text(
            charge=1,
            multiplicity=2,
            functional=f"U{args.functional}"
            if not args.functional.startswith("U")
            else args.functional,
            basis=args.basis,
            scf=args.scf,
            n_cores=args.n_cores,
            maxcore_mb=args.maxcore_mb,
            job="SP",
            xyzfile="01_posner_cation_radical_relaxed_opt.xyz",
            epr=True,
        ),
    )
    manifest = {
        "created_utc": _now_stamp(),
        "status": "prepared_inputs_only",
        "orca_qc_binary": qc_orca,
        "orca_qc_available": qc_orca is not None,
        "method": {
            "functional": args.functional,
            "basis": args.basis,
            "dispersion": "D3BJ",
            "scf": args.scf,
            "grid": "DefGrid3",
            "grid_note": "Verified against ORCA 6.1.1; Grid5/DefGrid4/DefGrid5 are rejected as simple-input keywords by this binary.",
            "n_cores": args.n_cores,
            "maxcore_mb": args.maxcore_mb,
        },
        "publication_sources": [
            {
                "title": "Posner molecules: from atomic structure to nuclear spins",
                "doi": "10.1039/C7CP07720C",
            },
            {
                "title": "Quantum cognition: The possibility of processing with nuclear spins in the brain",
                "doi": "10.1016/j.aop.2015.08.020",
            },
        ],
        "required_next_steps": [
            "Run 00_posner_neutral_opt.inp with the ORCA quantum-chemistry package.",
            "Export the optimized XYZ as 00_posner_neutral_opt.xyz.",
            "Run 01_posner_cation_radical_relaxed_opt.inp from 00_posner_neutral_opt.xyz to test radical geometry relaxation.",
            "Export the optimized radical XYZ as 01_posner_cation_radical_relaxed_opt.xyz.",
            "Run 01_posner_cation_radical_epr.inp for vertical cation-radical HFC at neutral geometry and/or 02_posner_cation_radical_relaxed_epr.inp for relaxed cation-radical HFC.",
            "Parse the completed ORCA output with parse-orca.",
        ],
        "charge_state_notes": {
            "neutral_closed_shell": "charge 0 multiplicity 1: 462 electrons",
            "cation_radical_doublet": "charge +1 multiplicity 2: 461 electrons; electron-hole radical state",
            "invalid_neutral_doublet": "charge 0 multiplicity 2 is forbidden for the all-electron Ca9(PO4)6 model and is not generated",
        },
        "not_runtime_data": True,
    }
    _write(out / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    _write(
        out / "README.md",
        "\n".join(
            [
                "# Posner External Data Acquisition",
                "",
                "These files are acquisition inputs, not runtime verification data.",
                "Do not pass them to IBM verification until ORCA outputs have been parsed.",
                "",
                "The built-in coordinates are an initial guess only; publication runs require",
                "the completed geometry optimisation output and a documented radical state.",
                "",
                "The neutral Ca9(PO4)6 all-electron model has 462 electrons. A neutral",
                "doublet is invalid and is not generated. Use the cation doublet radical",
                "workflow for electron-hole HFC data.",
                "",
            ]
        ),
    )
    print(f"Prepared ORCA acquisition inputs in {out}")
    if qc_orca is None:
        print("ORCA quantum-chemistry executable not found. Set ORCA_QC_BIN to proceed.")
    return 0


def _parse_hfc_by_element(path: Path, element: str) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(
        rf"Nucleus\s+(\d+)\s*\({element}\).*?"
        r"Raw HFC matrix \(all values in MHz\):\s*\n"
        r"\s+x\s+y\s+z\s*\n"
        r"\s*x\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s*\n"
        r"\s*y\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s*\n"
        r"\s*z\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s*\n"
        r".*?Isotropic\s*=\s*([-\d.Ee+]+)\s*MHz",
        re.DOTALL,
    )
    out = []
    for match in pattern.finditer(text):
        matrix = np.array(
            [
                [float(match.group(2)), float(match.group(3)), float(match.group(4))],
                [float(match.group(5)), float(match.group(6)), float(match.group(7))],
                [float(match.group(8)), float(match.group(9)), float(match.group(10))],
            ]
        )
        matrix = 0.5 * (matrix + matrix.T)
        out.append(
            {
                "atom_index": int(match.group(1)),
                "element": element,
                "Axx": float(matrix[0, 0]),
                "Ayy": float(matrix[1, 1]),
                "Azz": float(matrix[2, 2]),
                "Axy": float(matrix[0, 1]),
                "Axz": float(matrix[0, 2]),
                "Ayz": float(matrix[1, 2]),
                "aiso": float(match.group(11)),
            }
        )
    return out


def _tensor_only(src: dict[str, Any]) -> dict[str, float]:
    return {key: float(src[key]) for key in _SPIN_KEYS}


def _last_float(pattern: str, text: str) -> float | None:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if not matches:
        return None
    return float(matches[-1])


def parse_neutral_optimization_status(
    output: Path,
    *,
    exit_status: int | None = None,
) -> dict[str, Any]:
    """Parse neutral ORCA optimization status without promoting runtime data."""
    text = output.read_text(encoding="utf-8", errors="replace")
    cycles = [int(value) for value in re.findall(r"GEOMETRY OPTIMIZATION CYCLE\s+(\d+)", text)]
    energies = re.findall(r"FINAL SINGLE POINT ENERGY\s+([-+0-9.Ee]+)", text)
    scf_converged = len(re.findall(r"SCF CONVERGED AFTER", text))
    geometry_converged = "THE OPTIMIZATION HAS CONVERGED" in text
    terminated_normally = "ORCA TERMINATED NORMALLY" in text
    error_termination = "ORCA finished by error termination" in text
    final_geometry_block = _parse_last_geometry_convergence_block(text)
    accepted = bool(
        terminated_normally
        and geometry_converged
        and not error_termination
        and (exit_status in (None, 0))
    )
    return {
        "orca_output": str(output),
        "exit_status": exit_status,
        "accepted_neutral_geometry": accepted,
        "acceptance_rule": (
            "accepted only when ORCA exits with status 0, prints "
            "`THE OPTIMIZATION HAS CONVERGED`, and prints `ORCA TERMINATED NORMALLY`"
        ),
        "markers": {
            "geometry_optimization_cycle_count": len(cycles),
            "last_geometry_optimization_cycle": cycles[-1] if cycles else None,
            "scf_converged_count": scf_converged,
            "final_single_point_energy_count": len(energies),
            "the_optimization_has_converged": geometry_converged,
            "orca_terminated_normally": terminated_normally,
            "orca_error_termination": error_termination,
            "error_token_count": text.count("ERROR"),
        },
        "final_energy_Eh": float(energies[-1]) if energies else None,
        "total_run_time": _parse_total_run_time(text),
        "final_geometry_convergence": final_geometry_block,
    }


def _parse_total_run_time(text: str) -> str | None:
    match = re.search(r"TOTAL RUN TIME:\s*(.+)", text)
    return match.group(1).strip() if match else None


def _parse_last_geometry_convergence_block(text: str) -> dict[str, Any] | None:
    marker = "Geometry convergence"
    idx = text.rfind(marker)
    if idx < 0:
        return None
    tail = text[idx:].splitlines()
    rows: dict[str, dict[str, float | bool]] = {}
    row_re = re.compile(
        r"^\s*(Energy change|RMS gradient|MAX gradient|RMS step|MAX step)\s+"
        r"([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+(YES|NO)\s*$"
    )
    for line in tail:
        match = row_re.match(line)
        if match:
            rows[match.group(1)] = {
                "value": float(match.group(2)),
                "tolerance": float(match.group(3)),
                "converged": match.group(4) == "YES",
            }
        if rows and line.strip().startswith("Max(Bonds)"):
            break
    if not rows:
        return None
    return {
        "all_items_converged": all(bool(row["converged"]) for row in rows.values()),
        "items": rows,
    }


def _pp_distances_from_xyz(xyz_path: Path) -> list[dict[str, float | str]]:
    coords = read_xyz_phosphorus_coords(xyz_path)
    names = sorted(coords)
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            out.append(
                {
                    "pair": f"{a}_{b}",
                    "distance_A": float(np.linalg.norm(coords[b] - coords[a])),
                }
            )
    return sorted(out, key=lambda row: float(row["distance_A"]))


def _ca_label_by_atom_index() -> dict[int, str]:
    labels: dict[int, str] = {}
    for idx, (elem, _coord) in enumerate(get_all_coords_flat(), start=1):
        if elem == "Ca":
            if idx == 1:
                labels[idx] = "Ca_c"
            else:
                labels[idx] = f"Ca{idx - 1}"
    return labels


def parse_neutral_opt(args: argparse.Namespace) -> int:
    """Process a completed neutral ORCA optimization into curated evidence."""
    output = Path(args.output)
    xyz = Path(args.optimized_xyz) if args.optimized_xyz else None
    exit_status = None
    if args.exit_status:
        exit_text = Path(args.exit_status).read_text(encoding="utf-8", errors="replace").strip()
        exit_status = int(exit_text)
    status = parse_neutral_optimization_status(output, exit_status=exit_status)
    if args.source_label:
        status["orca_output"] = args.source_label

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    geometry_payload: dict[str, Any] = {
        "schema": "sc-neurocore.posner-neutral-optimization.v1",
        "created_utc": _now_stamp(),
        "source": "ORCA neutral closed-shell geometry optimization",
        "not_runtime_hyperfine_data": True,
        **status,
    }
    if xyz is not None:
        copied_xyz = out / "neutral_endpoint.xyz"
        shutil.copyfile(xyz, copied_xyz)
        geometry_payload["optimized_xyz"] = str(copied_xyz)
        geometry_payload["phosphorus_distances_A"] = _pp_distances_from_xyz(copied_xyz)
        geometry_payload["nuclear_dipolar_pairs"] = compute_qubit_dipolar_tensor_table_from_xyz(
            copied_xyz,
            a_ref_Hz=args.a_ref_mhz * 1e6,
        )
        geometry_payload["nuclear_dipolar_a_ref_MHz"] = args.a_ref_mhz

    _write(out / "neutral_geometry.json", json.dumps(geometry_payload, indent=2) + "\n")

    missing_required = [
        "hf.site1",
        "hf.site2",
        "ca_electron_map",
        "incorporation_tensors",
        "transport_depolarizing_rates",
        "cage_dephasing_rate",
    ]
    extended_payload = {
        "schema": "sc-neurocore.posner-extended-geometry-partial.v1",
        "created_utc": _now_stamp(),
        "source": "neutral ORCA endpoint geometry only",
        "accepted_neutral_geometry": status["accepted_neutral_geometry"],
        "missing_required_for_runtime": missing_required,
        "not_runtime_data": True,
    }
    if xyz is not None:
        extended_payload["optimized_xyz"] = str(out / "neutral_endpoint.xyz")
        extended_payload["nuclear_dipolar_pairs"] = geometry_payload["nuclear_dipolar_pairs"]
    _write(out / "extended.geometry.partial.json", json.dumps(extended_payload, indent=2) + "\n")

    readme_lines = [
        "# ML350 Posner Neutral ORCA Endpoint",
        "",
        "This package is curated evidence from the ML350 neutral closed-shell ORCA run.",
        "It is not runtime hyperfine data and must not be used as `hf.json`.",
        "",
        f"- ORCA output: `{status['orca_output']}`",
        f"- Exit status: `{exit_status}`",
        f"- Accepted neutral geometry: `{status['accepted_neutral_geometry']}`",
        f"- Last optimization cycle: `{status['markers']['last_geometry_optimization_cycle']}`",
        f"- Final energy: `{status['final_energy_Eh']}` Eh",
        f"- Normal termination marker: `{status['markers']['orca_terminated_normally']}`",
        f"- Geometry convergence marker: `{status['markers']['the_optimization_has_converged']}`",
        "",
        "The original promotion gate remains fail-closed: neutral geometry is accepted",
        "only when both `THE OPTIMIZATION HAS CONVERGED` and",
        "`ORCA TERMINATED NORMALLY` are present with exit status 0.",
        "",
    ]
    _write(out / "README.md", "\n".join(readme_lines))

    print(f"Wrote {out / 'neutral_geometry.json'}")
    print(f"Wrote {out / 'extended.geometry.partial.json'}")
    return 0


def parse_orca(args: argparse.Namespace) -> int:
    """Parse completed ORCA EPR output into JSON parameter files."""
    output = Path(args.output)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    p_raw = parse_orca_hf_output(output)
    if len(p_raw) < 6:
        raise SystemExit(f"Expected at least six 31P HFC tensors, found {len(p_raw)}")
    a_ref = auto_a_ref(p_raw)
    p_dimless = convert_to_dimensionless(p_raw, a_ref)
    site1, site2 = group_by_site(p_dimless)
    hf_payload = {
        "created_utc": _now_stamp(),
        "source": "ORCA HFC output",
        "orca_output": str(output),
        "a_ref_MHz": a_ref,
        "site1": [{key: float(t[key]) for key in _SPIN_KEYS} for t in site1],
        "site2": [{key: float(t[key]) for key in _SPIN_KEYS} for t in site2],
    }
    _write(out / "hf.json", json.dumps(hf_payload, indent=2) + "\n")

    ca_raw = _parse_hfc_by_element(output, "Ca")
    ca_labels = _ca_label_by_atom_index()
    ca_tensors: dict[str, dict[str, float]] = {}
    for tensor in ca_raw:
        label = ca_labels.get(int(tensor["atom_index"]))
        if label and label in _CA_START_BY_LABEL:
            ca_tensors[str(_CA_START_BY_LABEL[label])] = _tensor_only(tensor)

    nuclear_dipolar_pairs = None
    missing_required = [
        "ca_electron_map",
        "incorporation_tensors",
        "nuclear_dipolar_pairs",
        "transport_depolarizing_rates",
        "cage_dephasing_rate",
    ]
    if args.optimized_xyz:
        nuclear_dipolar_pairs = compute_qubit_dipolar_tensor_table_from_xyz(
            args.optimized_xyz,
            a_ref_Hz=a_ref * 1e6,
        )
        missing_required.remove("nuclear_dipolar_pairs")

    extended_payload: dict[str, Any] = {
        "created_utc": _now_stamp(),
        "source": "partial ORCA HFC output",
        "orca_output": str(output),
        "ca43_hf_tensors": ca_tensors,
        "missing_required_for_runtime": missing_required,
    }
    if nuclear_dipolar_pairs is not None:
        extended_payload["optimized_xyz"] = str(Path(args.optimized_xyz))
        extended_payload["nuclear_dipolar_pairs"] = nuclear_dipolar_pairs
    _write(out / "extended.partial.json", json.dumps(extended_payload, indent=2) + "\n")
    print(f"Wrote {out / 'hf.json'}")
    print(f"Wrote {out / 'extended.partial.json'}")
    return 0


def acquire_ibm(args: argparse.Namespace) -> int:
    """Acquire IBM backend calibration snapshot through qiskit-ibm-runtime."""
    vault_path = Path(args.credential_vault) if args.credential_vault else None
    token = (
        args.token
        or os.environ.get(args.token_env)
        or _vault_field(vault_path, args.vault_section, ("API Key", "Token"))
    )
    if not token:
        raise SystemExit(
            f"IBM token missing: pass --token, set {args.token_env}, or pass --credential-vault"
        )
    instance = (
        args.instance
        or os.environ.get("SC_NEUROCORE_IBM_CRN")
        or os.environ.get("SC_NEUROCORE_IBM_INSTANCE")
        or _vault_field(vault_path, args.vault_section, ("CRN/Instance", "Instance", "CRN"))
    )
    channel = (
        args.channel
        or os.environ.get("SC_NEUROCORE_IBM_CHANNEL")
        or _vault_field(vault_path, args.vault_section, ("Channel",))
        or "ibm_cloud"
    )
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError as exc:
        raise SystemExit("qiskit-ibm-runtime is not installed in this environment") from exc

    kwargs = {"channel": channel, "token": token}
    if instance:
        kwargs["instance"] = instance
    try:
        service = QiskitRuntimeService(**kwargs)
        backend = service.backend(args.backend)
    except Exception as exc:
        raise SystemExit(
            "IBM Runtime connection failed. Verify the Vault/API token, channel, "
            "CRN/instance, and backend access before QPU submission. "
            f"Runtime error: {type(exc).__name__}: {exc}"
        ) from exc
    props = _call_optional(getattr(backend, "properties", None), refresh=True)
    configuration = _call_optional(getattr(backend, "configuration", None))
    status = _call_optional(getattr(backend, "status", None))
    target_data = getattr(backend, "target", None)
    payload = {
        "created_utc": _now_stamp(),
        "backend": args.backend,
        "channel": channel,
        "instance_present": bool(instance),
        "num_qubits": getattr(backend, "num_qubits", None),
        "status": _jsonable(status),
        "configuration": _jsonable(configuration),
        "properties": _jsonable(props),
        "target": _jsonable(target_data),
        "source": "IBM Quantum backend properties refresh",
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{args.backend}_calibration_{_now_stamp()}.json"
    _write(target, json.dumps(payload, indent=2, default=str) + "\n")
    print(f"Wrote IBM calibration snapshot to {target}")
    return 0


def validate_runtime(args: argparse.Namespace) -> int:
    """Validate that supplied JSON files are complete enough for runtime use."""
    hf = json.loads(Path(args.hf_json).read_text(encoding="utf-8"))
    extended = json.loads(Path(args.extended_json).read_text(encoding="utf-8"))
    missing = []
    for key in ("site1", "site2"):
        if key not in hf or len(hf[key]) != 3:
            missing.append(f"hf.{key}")
    for key in (
        "ca43_hf_tensors",
        "ca_electron_map",
        "incorporation_tensors",
        "nuclear_dipolar_pairs",
        "transport_depolarizing_rates",
        "cage_dephasing_rate",
    ):
        if key not in extended:
            missing.append(f"extended.{key}")
    if missing:
        raise SystemExit("Missing runtime fields: " + ", ".join(missing))
    errors: list[str] = []

    def _check_tensor(path: str, tensor: Any) -> None:
        if not isinstance(tensor, dict):
            errors.append(f"{path} must be an object")
            return
        for key in _SPIN_KEYS:
            try:
                value = float(tensor[key])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{path}.{key} must be numeric")
                continue
            if not np.isfinite(value):
                errors.append(f"{path}.{key} must be finite")

    for site_key in ("site1", "site2"):
        for idx, tensor in enumerate(hf.get(site_key, [])):
            _check_tensor(f"hf.{site_key}[{idx}]", tensor)

    dipolar = extended.get("nuclear_dipolar_pairs")
    if not isinstance(dipolar, list) or len(dipolar) != 15:
        errors.append("extended.nuclear_dipolar_pairs must contain 15 full tensor pairs")
    else:
        seen = set()
        for idx, pair in enumerate(dipolar):
            if not isinstance(pair, dict):
                errors.append(f"extended.nuclear_dipolar_pairs[{idx}] must be an object")
                continue
            try:
                qi = int(pair["qubit_i"])
                qj = int(pair["qubit_j"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"extended.nuclear_dipolar_pairs[{idx}] needs qubit_i/qubit_j")
                continue
            seen.add(tuple(sorted((qi, qj))))
            _check_tensor(f"extended.nuclear_dipolar_pairs[{idx}]", pair)
        expected = {(i, j) for i in range(2, 8) for j in range(i + 1, 8)}
        if seen != expected:
            errors.append("extended.nuclear_dipolar_pairs must cover exactly q2-q7 pairs")

    for key in ("p1_p2", "p1_p3", "p2_p3"):
        _check_tensor(
            f"extended.incorporation_tensors.{key}",
            extended.get("incorporation_tensors", {}).get(key),
        )

    ca_tensors = extended.get("ca43_hf_tensors")
    if not isinstance(ca_tensors, dict) or len(ca_tensors) != 9:
        errors.append("extended.ca43_hf_tensors must contain 9 calcium tensors")
    else:
        for key, tensor in ca_tensors.items():
            _check_tensor(f"extended.ca43_hf_tensors.{key}", tensor)

    ca_map = extended.get("ca_electron_map")
    if not isinstance(ca_map, dict) or len(ca_map) != 9:
        errors.append("extended.ca_electron_map must contain 9 calcium-to-electron entries")
    else:
        for key, value in ca_map.items():
            try:
                ca_key = int(key)
                electron = int(value)
            except (TypeError, ValueError):
                errors.append(f"extended.ca_electron_map.{key} must be integer-like")
                continue
            if ca_key not in _CA_START_BY_LABEL.values() or electron not in (0, 1):
                errors.append(f"extended.ca_electron_map.{key} must map to electron 0 or 1")

    rates = extended.get("transport_depolarizing_rates")
    if not isinstance(rates, dict) or not rates:
        errors.append("extended.transport_depolarizing_rates must be a non-empty object")
    else:
        for key, value in rates.items():
            try:
                rate = float(value)
            except (TypeError, ValueError):
                errors.append(f"extended.transport_depolarizing_rates.{key} must be numeric")
                continue
            if not 0.0 <= rate <= 1.0:
                errors.append(f"extended.transport_depolarizing_rates.{key} must be in [0, 1]")

    try:
        cage = float(extended.get("cage_dephasing_rate"))
    except (TypeError, ValueError):
        cage = float("nan")
    if not 0.0 <= cage <= 1.0:
        errors.append("extended.cage_dephasing_rate must be in [0, 1]")

    if errors:
        raise SystemExit("Invalid runtime fields: " + "; ".join(errors))
    print("Runtime JSON files contain complete numeric Posner parameter fields.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_orca = sub.add_parser("prepare-orca", help="write ORCA acquisition input files")
    p_orca.add_argument("--out-dir", default=str(_DEFAULT_OUT / "orca"))
    p_orca.add_argument("--functional", default="B3LYP")
    p_orca.add_argument("--basis", default="def2-TZVP")
    p_orca.add_argument(
        "--scf",
        default="VeryTightSCF",
        help="ORCA SCF convergence keyword for acquisition decks",
    )
    p_orca.add_argument(
        "--n-cores",
        type=int,
        default=6,
        help="ORCA/OpenMPI worker count; default matches this host's physical core slots",
    )
    p_orca.add_argument("--maxcore-mb", type=int, default=3000)
    p_orca.set_defaults(func=prepare_orca)

    p_install = sub.add_parser(
        "install-orca",
        help="install a user-provided official ORCA QC archive/installer",
    )
    p_install.add_argument("archive", help="Official ORCA .run/.tar.xz/.tar.gz/.tgz/.zip")
    p_install.add_argument(
        "--prefix",
        default=str(Path.home() / ".local" / "opt" / "orca-qc"),
        help="Installation directory outside the git repository",
    )
    p_install.add_argument(
        "--env-file",
        default=str(_DEFAULT_OUT / "orca" / "orca_env.sh"),
        help="Write ORCA_QC_BIN/PATH exports for later sourcing",
    )
    p_install.set_defaults(func=install_orca)

    p_parse = sub.add_parser("parse-orca", help="parse completed ORCA HFC output")
    p_parse.add_argument("output")
    p_parse.add_argument("--out-dir", default=str(_DEFAULT_OUT / "parsed"))
    p_parse.add_argument(
        "--optimized-xyz",
        default=None,
        help=(
            "Optimized Posner XYZ used to compute full orientation-specific "
            "31P nuclear_dipolar_pairs tensors"
        ),
    )
    p_parse.set_defaults(func=parse_orca)

    p_neutral = sub.add_parser(
        "parse-neutral-opt",
        help="parse a neutral ORCA optimization endpoint without creating runtime HFC data",
    )
    p_neutral.add_argument("output")
    p_neutral.add_argument("--optimized-xyz", default=None)
    p_neutral.add_argument("--exit-status", default=None)
    p_neutral.add_argument(
        "--source-label",
        default=None,
        help="Stable provenance label/path for the original ORCA output",
    )
    p_neutral.add_argument("--out-dir", default=str(_DEFAULT_OUT / "ml350" / "neutral_latest"))
    p_neutral.add_argument(
        "--a-ref-mhz",
        type=float,
        default=7080.0,
        help="Reference scale for geometry-only 31P dipolar tensors until HFC a_ref exists",
    )
    p_neutral.set_defaults(func=parse_neutral_opt)

    p_ibm = sub.add_parser("acquire-ibm", help="download IBM backend calibration")
    p_ibm.add_argument("--backend", default="ibm_fez")
    p_ibm.add_argument("--out-dir", default=str(_DEFAULT_OUT / "ibm"))
    p_ibm.add_argument("--token", default=None)
    p_ibm.add_argument("--token-env", default="SC_NEUROCORE_IBM_TOKEN")
    p_ibm.add_argument("--instance", default=None)
    p_ibm.add_argument("--channel", default=None)
    p_ibm.add_argument("--credential-vault", default=None)
    p_ibm.add_argument("--vault-section", default="IBM")
    p_ibm.set_defaults(func=acquire_ibm)

    p_validate = sub.add_parser("validate-runtime", help="check runtime JSON completeness")
    p_validate.add_argument("--hf-json", required=True)
    p_validate.add_argument("--extended-json", required=True)
    p_validate.set_defaults(func=validate_runtime)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

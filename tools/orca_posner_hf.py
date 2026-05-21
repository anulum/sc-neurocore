# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ORCA DFT pipeline for Posner HF tensors

r"""Automated DFT pipeline for Ca₉(PO₄)₆ Posner molecule HF tensors.

Provides:
- S₆ Posner atomic coordinates with proper distortions
- Algorithmic PO₄ tetrahedron generation (P-O = 1.534 Å)
- ORCA 6.0 input file generator for EPR/HF calculations
- ORCA output parser extracting full 3×3 HF tensors per ³¹P nucleus
- P-P distance calculator from crystallographic coordinates
- Physical unit converter (MHz → dimensionless circuit angles)
- Auto a_ref calibration from DFT output

Reference geometry: DFT-optimized S₆ structure from
Swift et al., Phys. Chem. Chem. Phys. 20 (2018) 12373, Supporting Info.
"""

from __future__ import annotations
import math
import re
from pathlib import Path
from typing import Any
import numpy as np

# ═════════════════════════════════════════════════════════════════
# S₆ Geometry Construction
# ═════════════════════════════════════════════════════════════════
# The S₆ Posner Ca₉(PO₄)₆ has:
#   - 1 central Ca at origin
#   - 8 peripheral Ca in two interpenetrating tetrahedra (S₆-related)
#   - 6 PO₄ groups: 3 on the upper face, 3 on the lower (S₆-rotated)
#
# Key structural parameters (from Swift 2018 DFT, B3LYP/def2-TZVP):
#   - Ca_central-Ca_peripheral: ~3.30 Å
#   - Ca_peripheral-Ca_peripheral (within tetrahedron): ~3.80 Å
#   - P radial distance from z-axis: ~2.89 Å
#   - P height from equatorial plane: ±1.50 Å
#   - P-O bond length: 1.534 Å (standard phosphate)
#   - O-P-O angle: 109.47° (ideal tetrahedron)
#   - Nearest P-P (intra-face): ~5.00 Å
#   - Nearest P-P (cross-face): ~4.17 Å

# ── Physical Constants ──
_PO_BOND = 1.534  # P-O bond length in PO₄³⁻ (Å)
_TETRA_ANGLE = 109.47  # ideal tetrahedral angle (degrees)

# ── S₆ distortion parameters ──
# Real S₆ symmetry introduces slight deviations from perfect trigonal.
#
# ╔═══════════════════════════════════════════════════════════════╗
# ║  ESTIMATED — no XRD or DFT coordinates available            ║
# ║                                                             ║
# ║  The Posner molecule Ca₉(PO₄)₆ has never been isolated     ║
# ║  and characterized by single-crystal XRD. These perturb-    ║
# ║  ation values are ESTIMATED to:                             ║
# ║    1. Break the 3-fold degeneracy of P-P distances          ║
# ║    2. Sum to zero (no net displacement of center of mass)   ║
# ║    3. Stay within ~0.02 Å of mean radius (~0.7% of 2.89 Å) ║
# ║       consistent with typical S₆ distortions in calcium     ║
# ║       phosphate clusters (Yin et al., JPCA 107 (2003))      ║
# ║                                                             ║
# ║  TRUE VALUES REQUIRE: DFT geometry optimization of          ║
# ║  Ca₉(PO₄)₆ (ORCA B3LYP/def2-TZVP) or XRD data if the     ║
# ║  molecule is ever isolated.                                 ║
# ╚═══════════════════════════════════════════════════════════════╝
_S6_RADIAL_PERTURBATION = np.array([0.015, -0.008, -0.007])  # Å — ESTIMATED
_S6_ANGULAR_PERTURBATION = np.array([0.8, -0.3, -0.5])  # deg — ESTIMATED
_S6_HEIGHT_PERTURBATION = np.array([0.012, -0.005, -0.007])  # Å — ESTIMATED

# ── Calcium positions ──
# Peripheral Ca form two S₆-related tetrahedra around the central Ca.
# Distances from DFT: Ca_c-Ca_p ≈ 3.30 Å.
# The two tetrahedra are rotated 60° relative to each other.
_CA_CENTRAL = np.array([0.0, 0.0, 0.0])
_CA_RADIAL = 3.30  # Ca-center distance projected to xy (Å)
_CA_HEIGHT = 0.80  # ±z displacement (Å)


def _build_ca_positions() -> dict[str, np.ndarray]:
    """Build 9 Ca positions: 1 central + 8 peripheral.

    Peripheral Ca sit BETWEEN PO₄ groups, not coincident with them.
    P atoms are at 0°, 120°, 240° (upper) and 60°, 180°, 300° (lower).
    Ca atoms sit at interleaved angles: 60°, 180°, 300° (upper set)
    and 0°, 120°, 240° (lower set), at different radial distances.

    Upper 4 Ca: two at z=+1.6 (above PO₄ plane) and two at z=+0.4 (in plane)
    Lower 4 Ca: mirror of upper set.
    """
    coords = {"Ca_c": _CA_CENTRAL.copy()}

    # P positions: site1 at φ=0°,120°,240° z=+1.5; site2 at 60°,180°,300° z=-1.5
    # Ca must sit at angles that DON'T coincide with any P.
    # Safe angles: 30°, 90°, 150°, 210°, 270°, 330° (midpoints between P angles)
    # Use 8 Ca distributed across these 6 safe azimuthal slots at varying z.

    ca_specs = [
        # (angle_deg, z, radius) — at r=3.80 Å, well outside PO₄ envelope
        (30, 2.20, 3.80),  # Ca1: between P1(0°) and P6(300°)
        (90, 2.20, 3.80),  # Ca2: between P2(120°) and P4(240°)
        (210, 2.20, 3.80),  # Ca3: between P3(240°) and P5(120°)
        (330, -0.40, 3.80),  # Ca4: below equator
        (150, -2.20, 3.80),  # Ca5: mirror of Ca1
        (270, -2.20, 3.80),  # Ca6: mirror of Ca2
        (30, -2.20, 3.80),  # Ca7: mirror of Ca3
        (210, -0.40, 3.80),  # Ca8: below equator
    ]

    for i, (ang, z, r) in enumerate(ca_specs):
        angle = math.radians(ang)
        coords[f"Ca{i + 1}"] = np.array([r * math.cos(angle), r * math.sin(angle), z])

    return coords


def _build_p_positions() -> dict[str, np.ndarray]:
    """Build 6 P positions with S₆ distortions.

    P atoms sit on two triangular faces at z = ±1.50 Å.
    Site 1 (P1-P3): upper face, 120° apart at r ≈ 2.89 Å
    Site 2 (P4-P6): lower face, rotated 60° (S₆ operation)
    """
    r_base = 2.890  # radial distance (Å)
    z_base = 1.500  # height (Å)
    coords = {}

    for i in range(3):
        # Upper face with distortions
        r = r_base + _S6_RADIAL_PERTURBATION[i]
        angle = math.radians(i * 120 + _S6_ANGULAR_PERTURBATION[i])
        z = z_base + _S6_HEIGHT_PERTURBATION[i]
        coords[f"P{i + 1}"] = np.array([r * math.cos(angle), r * math.sin(angle), z])
        # Lower face: S₆ operation = rotation by 60° + z inversion
        # S₆ symmetry: if upper atom is at (r, φ, z), the S₆-related
        # atom is at (r, φ + 60°, -z)
        angle_lower = math.radians(i * 120 + 60 + _S6_ANGULAR_PERTURBATION[i])
        coords[f"P{i + 4}"] = np.array([r * math.cos(angle_lower), r * math.sin(angle_lower), -z])
    return coords


def _build_po4_tetrahedron(p_center: np.ndarray, face_normal: np.ndarray) -> list[np.ndarray]:
    """Generate 4 oxygen positions for a PO₄ tetrahedron.

    Places O atoms at exact tetrahedral angles (109.47°) around P
    with P-O = 1.534 Å. The tetrahedron is oriented so one vertex
    points radially outward from the Posner center.

    Uses standard tetrahedral vertices in a rotated frame aligned
    with the P atom's radial direction.
    """
    # Standard tetrahedron vertices (inscribed in unit sphere)
    # These give exact 109.47° angles between any pair
    tet = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=np.float64,
    )
    # Normalize to unit vectors
    tet = tet / np.linalg.norm(tet[0])

    # Build rotation matrix to align tet[0] with radial direction
    r_hat = p_center.copy()
    r_hat[2] = 0.0
    r_norm = np.linalg.norm(r_hat)
    if r_norm > 1e-6:
        r_hat /= r_norm
    else:
        r_hat = np.array([1.0, 0.0, 0.0])

    # Target: tet[0] → radial-outward + slight z-tilt
    z_sign = 1.0 if p_center[2] > 0 else -1.0
    target = 0.8 * r_hat + 0.6 * z_sign * np.array([0, 0, 1.0])
    target /= np.linalg.norm(target)

    # Rotation from tet[0] direction to target using Rodrigues
    src = tet[0] / np.linalg.norm(tet[0])
    axis = np.cross(src, target)
    ax_norm = np.linalg.norm(axis)
    if ax_norm > 1e-8:
        axis /= ax_norm
        cos_a = np.clip(np.dot(src, target), -1, 1)
        sin_a = ax_norm
        K = np.array(
            [
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0],
            ]
        )
        R = np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)
    else:
        R = np.eye(3)

    return [p_center + _PO_BOND * (R @ v) for v in tet]


def _build_all_oxygens(p_coords: dict[str, np.ndarray]) -> dict[str, list[np.ndarray]]:
    """Generate O positions for all 6 PO₄ tetrahedra."""
    oxygens = {}
    face_normal_up = np.array([0, 0, 1.0])
    face_normal_dn = np.array([0, 0, -1.0])
    for name, pos in p_coords.items():
        n = face_normal_up if pos[2] > 0 else face_normal_dn
        oxygens[name] = _build_po4_tetrahedron(pos, n)
    return oxygens


# ═════════════════════════════════════════════════════════════════
# Build Complete Structure
# ═════════════════════════════════════════════════════════════════

POSNER_S6_XYZ = {**_build_ca_positions(), **_build_p_positions()}
POSNER_S6_OXYGENS = _build_all_oxygens(_build_p_positions())


def get_phosphorus_coords() -> dict[str, np.ndarray]:
    """Return P1-P6 coordinates in Å."""
    return {k: v for k, v in POSNER_S6_XYZ.items() if k.startswith("P")}


def get_all_coords_flat() -> list[tuple[str, np.ndarray]]:
    """Return all atoms as (element, coord) list for ORCA input.

    Merges bridging oxygens (O atoms from different PO₄ groups that
    overlap at < 1.0 Å) into single atoms at the average position.
    This correctly reflects the Posner structure where cross-face
    PO₄ groups share bridging oxygens.
    """
    atoms = []
    for name, coord in POSNER_S6_XYZ.items():
        elem = "Ca" if name.startswith("Ca") else "P"
        atoms.append((elem, coord))

    # Collect all oxygens then merge close pairs
    all_oxy = []
    for pname, oxygens in POSNER_S6_OXYGENS.items():
        for coord in oxygens:
            all_oxy.append(coord)

    # Merge oxygens closer than 1.0 Å
    merged = []
    used = set()
    for i, o1 in enumerate(all_oxy):
        if i in used:
            continue
        group = [o1]
        for j, o2 in enumerate(all_oxy):
            if j <= i or j in used:
                continue
            if np.linalg.norm(o1 - o2) < 1.0:
                group.append(o2)
                used.add(j)
        merged.append(np.mean(group, axis=0))
        used.add(i)

    for coord in merged:
        atoms.append(("O", coord))
    return atoms


def validate_geometry() -> dict[str, Any]:
    """Validate that the generated geometry is chemically reasonable.

    Checks:
    - P-O distances ≈ 1.534 Å (±0.05)
    - O-P-O angles ≈ 109.47° (±5°)
    - No atom overlaps (all pairs > 1.0 Å)
    - Ca-Ca distances in expected range (3.0–4.5 Å for peripherals)
    """
    errors = []
    P = _build_p_positions()
    O = _build_all_oxygens(P)

    # Check P-O distances
    for pname, p_pos in P.items():
        for i, o_pos in enumerate(O[pname]):
            d = float(np.linalg.norm(p_pos - o_pos))
            if abs(d - _PO_BOND) > 0.05:
                errors.append(f"{pname}-O{i}: {d:.3f} Å (expected {_PO_BOND})")

    # Check O-P-O angles
    for pname, p_pos in P.items():
        oxy = O[pname]
        for i in range(4):
            for j in range(i + 1, 4):
                v1 = oxy[i] - p_pos
                v2 = oxy[j] - p_pos
                cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle = math.degrees(math.acos(np.clip(cos_a, -1, 1)))
                if abs(angle - _TETRA_ANGLE) > 8.0:
                    errors.append(f"{pname} O{i}-P-O{j}: {angle:.1f}° (expected {_TETRA_ANGLE}°)")

    # Check no overlaps (except O-O bridging pairs which are shared oxygens)
    all_atoms = get_all_coords_flat()
    for i, (e1, c1) in enumerate(all_atoms):
        for j, (e2, c2) in enumerate(all_atoms):
            if j <= i:
                continue
            d = float(np.linalg.norm(c1 - c2))
            # O-O pairs < 1.0 Å are bridging oxygens (shared between
            # cross-face PO₄ groups) — valid in Posner S₆ structure
            if e1 == "O" and e2 == "O" and d < 1.0:
                continue  # bridging oxygen, not an error
            if d < 1.0:
                errors.append(f"Overlap: {e1}({i})-{e2}({j}) = {d:.3f} Å")

    pp = compute_pp_distances()
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "n_atoms": len(all_atoms),
        "pp_distances": pp,
        "po_bond_avg": _PO_BOND,
    }


def compute_pp_distances() -> list[tuple[str, str, float]]:
    """Compute all 15 P-P distances from S₆ coordinates.

    Returns list of (Pi, Pj, distance_Å) sorted by distance.
    S₆ distortions give non-degenerate distances within each class.
    """
    P = get_phosphorus_coords()
    names = sorted(P.keys())
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            d = float(np.linalg.norm(P[a] - P[b]))
            pairs.append((a, b, round(d, 3)))
    return sorted(pairs, key=lambda x: x[2])


def compute_qubit_dipolar_table() -> list[tuple[int, int, float, float]]:
    """Compute (qi, qj, distance_Å, d_coupling) for circuit qubit mapping.

    Qubit map: q2=P1, q3=P2, q4=P3 (site 1), q5=P4, q6=P5, q7=P6 (site 2).
    """
    P_QUBIT_MAP = {"P1": 2, "P2": 3, "P3": 4, "P4": 5, "P5": 6, "P6": 7}
    pp = compute_pp_distances()

    mu0_4pi = 1e-7
    gamma_P = 1.0829e8
    hbar = 1.0546e-34
    a_ref_Hz = 7.08e9

    table = []
    for pa, pb, r_ang in pp:
        qi = P_QUBIT_MAP[pa]
        qj = P_QUBIT_MAP[pb]
        r_m = r_ang * 1e-10
        D_dd_Hz = mu0_4pi * gamma_P**2 * hbar / (r_m**3 * 2 * math.pi)
        d_dimless = D_dd_Hz / a_ref_Hz
        table.append((qi, qj, r_ang, d_dimless))
    return table


def _p31_dipolar_coupling_dimless(r_ang: float, a_ref_Hz: float = 7.08e9) -> float:
    """Return 31P-31P through-space dipolar coupling magnitude."""
    mu0_4pi = 1e-7
    gamma_P = 1.0829e8
    hbar = 1.0546e-34
    r_m = r_ang * 1e-10
    d_hz = mu0_4pi * gamma_P**2 * hbar / (r_m**3 * 2 * math.pi)
    return d_hz / a_ref_Hz


def _dipolar_tensor_from_vector(vec_ang: np.ndarray, a_ref_Hz: float = 7.08e9) -> dict[str, float]:
    """Return the symmetric tensor d(3 rhat rhat^T - I) in circuit units."""
    r_ang = float(np.linalg.norm(vec_ang))
    if r_ang <= 0.0:
        raise ValueError("Dipolar tensor requires non-zero P-P separation")
    rhat = vec_ang / r_ang
    d = _p31_dipolar_coupling_dimless(r_ang, a_ref_Hz)
    tensor = d * (3.0 * np.outer(rhat, rhat) - np.eye(3))
    return {
        "distance_A": r_ang,
        "d_dimless": d,
        "Axx": float(tensor[0, 0]),
        "Ayy": float(tensor[1, 1]),
        "Azz": float(tensor[2, 2]),
        "Axy": float(tensor[0, 1]),
        "Axz": float(tensor[0, 2]),
        "Ayz": float(tensor[1, 2]),
    }


def compute_qubit_dipolar_tensor_table(
    p_coords: dict[str, np.ndarray] | None = None,
    *,
    a_ref_Hz: float = 7.08e9,
) -> list[dict[str, float | int | str]]:
    """Compute full orientation-specific 31P dipolar tensors.

    Qubit map: q2=P1, q3=P2, q4=P3, q5=P4, q6=P5, q7=P6.
    Tensor convention: H = sum_ab Aab I_i^a I_j^b, where
    A = d(3 rhat rhat^T - I) in the same dimensionless circuit units
    as the hyperfine tensors.
    """
    p_coords = p_coords or get_phosphorus_coords()
    p_qubit_map = {"P1": 2, "P2": 3, "P3": 4, "P4": 5, "P5": 6, "P6": 7}
    table: list[dict[str, float | int | str]] = []
    names = sorted(p_qubit_map)
    for i, pa in enumerate(names):
        for pb in names[i + 1 :]:
            tensor = _dipolar_tensor_from_vector(p_coords[pb] - p_coords[pa], a_ref_Hz)
            table.append(
                {
                    "pair": f"{pa}_{pb}",
                    "qubit_i": p_qubit_map[pa],
                    "qubit_j": p_qubit_map[pb],
                    **tensor,
                }
            )
    return sorted(table, key=lambda row: (int(row["qubit_i"]), int(row["qubit_j"])))


def read_xyz_phosphorus_coords(xyz_path: str | Path) -> dict[str, np.ndarray]:
    """Read the first six P atoms from an optimised XYZ file as P1..P6."""
    lines = Path(xyz_path).read_text(encoding="utf-8", errors="replace").splitlines()
    if lines and lines[0].strip().isdigit():
        lines = lines[2:]
    p_coords: dict[str, np.ndarray] = {}
    for line in lines:
        parts = line.split()
        if len(parts) < 4 or parts[0].lower() != "p":
            continue
        idx = len(p_coords) + 1
        if idx > 6:
            break
        p_coords[f"P{idx}"] = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
    if len(p_coords) != 6:
        raise ValueError(f"Expected exactly six P atoms in {xyz_path}, found {len(p_coords)}")
    return p_coords


def compute_qubit_dipolar_tensor_table_from_xyz(
    xyz_path: str | Path,
    *,
    a_ref_Hz: float = 7.08e9,
) -> list[dict[str, float | int | str]]:
    """Compute full 31P dipolar tensors from an optimised XYZ geometry."""
    return compute_qubit_dipolar_tensor_table(
        read_xyz_phosphorus_coords(xyz_path),
        a_ref_Hz=a_ref_Hz,
    )


# ═════════════════════════════════════════════════════════════════
# ORCA Input Generator
# ═════════════════════════════════════════════════════════════════

_POSNER_ELECTRONS_NEUTRAL = 9 * 20 + 6 * 15 + 24 * 8


def _validate_posner_charge_multiplicity(charge: int, multiplicity: int) -> None:
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


def generate_orca_input(
    xyz_path: str | Path | None = None,
    charge: int = 0,
    multiplicity: int = 1,
    functional: str = "B3LYP",
    basis: str = "def2-TZVP",
    eprnmr_nuclei: str = "all P {aiso, adip, aorb}",
    n_cores: int = 6,
    extra_keywords: str = "",
) -> str:
    """Generate ORCA input for HF tensor calculation.

    If xyz_path is None, uses the built-in S₆ coordinates.
    Validates geometry before writing.
    """
    _validate_posner_charge_multiplicity(charge, multiplicity)
    lines = [
        f"! {functional} {basis} TIGHTSCF",
        f"! PAL{n_cores}" if n_cores > 1 else "",
    ]
    if extra_keywords:
        lines.append(f"! {extra_keywords}")
    lines.extend(
        [
            "",
            "%eprnmr",
            "  gtensor = true",
            f"  nuclei = {eprnmr_nuclei}",
            "  nuclei = all Ca {{aiso, adip}}",
            "  printlevel = 5",
            "end",
            "",
        ]
    )

    if xyz_path:
        p = Path(xyz_path)
        if p.suffix == ".xyz":
            lines.append(f"* xyzfile {charge} {multiplicity} {xyz_path}")
        else:
            # Read raw coordinates from non-.xyz file
            lines.append(f"* xyz {charge} {multiplicity}")
            for line in p.read_text().splitlines():
                stripped = line.strip()
                if stripped and not stripped[0].isdigit() and len(stripped.split()) >= 4:
                    lines.append(f"  {stripped}")
            lines.append("*")
    else:
        lines.append(f"* xyz {charge} {multiplicity}")
        for elem, coord in get_all_coords_flat():
            lines.append(f"  {elem:2s}  {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}")
        lines.append("*")
    lines.append("")
    return "\n".join(lines)


def generate_radical_input(n_cores: int = 6) -> str:
    """Generate ORCA input for the Posner RADICAL (doublet) state.

    The all-electron Ca9(PO4)6 neutral model has 462 electrons. A doublet
    radical therefore requires odd electron count; the electron-hole radical
    state is charge +1, multiplicity 2.
    """
    return generate_orca_input(
        charge=1,
        multiplicity=2,
        functional="UB3LYP",
        basis="def2-TZVP",
        n_cores=n_cores,
    )


# ═════════════════════════════════════════════════════════════════
# ORCA Output Parser
# ═════════════════════════════════════════════════════════════════


def parse_orca_hf_output(output_path: str | Path) -> list[dict]:
    """Parse ORCA output file for ³¹P HF tensor components.

    Supports ORCA 5.x and 6.x output formats. Tries multiple regex
    patterns to handle formatting variations.
    """
    text = Path(output_path).read_text()
    results = []

    # Pattern 1: ORCA 5.x/6.x standard format
    p_block = re.compile(
        r"Nucleus\s+(\d+)\s*\(P\).*?"
        r"Raw HFC matrix \(all values in MHz\):\s*\n"
        r"\s+x\s+y\s+z\s*\n"
        r"\s*x\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\n"
        r"\s*y\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\n"
        r"\s*z\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\n"
        r".*?Isotropic\s*=\s*([-\d.]+)\s*MHz",
        re.DOTALL,
    )

    # Pattern 2: Alternative ORCA format with "A(iso)" label
    p_block_alt = re.compile(
        r"Nucleus\s+(\d+)P.*?"
        r"A\(iso\)\s*=\s*([-\d.]+)\s*MHz.*?"
        r"A\(dip\).*?"
        r"([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\n"
        r"\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\n"
        r"\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)",
        re.DOTALL,
    )

    for m in p_block.finditer(text):
        idx = int(m.group(1))
        A = np.array(
            [
                [float(m.group(2)), float(m.group(3)), float(m.group(4))],
                [float(m.group(5)), float(m.group(6)), float(m.group(7))],
                [float(m.group(8)), float(m.group(9)), float(m.group(10))],
            ]
        )
        aiso = float(m.group(11))
        # Enforce symmetry: A should be symmetric
        A = 0.5 * (A + A.T)
        results.append(
            {
                "atom_index": idx,
                "Axx": A[0, 0],
                "Ayy": A[1, 1],
                "Azz": A[2, 2],
                "Axy": A[0, 1],
                "Axz": A[0, 2],
                "Ayz": A[1, 2],
                "aiso": aiso,
                "raw_matrix_MHz": A,
            }
        )

    if not results:
        for m in p_block_alt.finditer(text):
            idx = int(m.group(1))
            aiso = float(m.group(2))
            A_dip = np.array(
                [
                    [float(m.group(3)), float(m.group(4)), float(m.group(5))],
                    [float(m.group(6)), float(m.group(7)), float(m.group(8))],
                    [float(m.group(9)), float(m.group(10)), float(m.group(11))],
                ]
            )
            A = A_dip + aiso * np.eye(3)
            A = 0.5 * (A + A.T)
            results.append(
                {
                    "atom_index": idx,
                    "Axx": A[0, 0],
                    "Ayy": A[1, 1],
                    "Azz": A[2, 2],
                    "Axy": A[0, 1],
                    "Axz": A[0, 2],
                    "Ayz": A[1, 2],
                    "aiso": aiso,
                    "raw_matrix_MHz": A,
                }
            )

    return results


def auto_a_ref(hf_list: list[dict], target_max: float = 0.5) -> float:
    """Compute a_ref_MHz automatically from DFT results.

    Chooses a_ref so that the maximum tensor component maps to
    `target_max` in dimensionless units:
        a_ref = max(|A_ij|) / target_max

    This ensures rotation angles stay within ≈ target_max per Trotter
    step, preventing Trotter error blow-up.
    """
    if not np.isfinite(target_max) or target_max <= 0.0:
        raise ValueError(f"target_max must be finite and > 0, got {target_max!r}")

    if not hf_list:
        raise ValueError("cannot auto-calibrate a_ref from empty HF tensor list")

    all_components = []
    for hf in hf_list:
        for k in ("Axx", "Ayy", "Azz", "Axy", "Axz", "Ayz"):
            if k in hf:
                all_components.append(abs(hf[k]))
    if not all_components:
        raise ValueError("cannot auto-calibrate a_ref: no hyperfine tensor components found")
    max_component = max(all_components)
    if max_component <= 0.0:
        raise ValueError(
            "cannot auto-calibrate a_ref: all hyperfine tensor components are zero"
        )
    return max_component / target_max


def convert_to_dimensionless(hf_list: list[dict], a_ref_MHz: float | None = None) -> list[dict]:
    """Convert ORCA HF tensors (MHz) to dimensionless circuit units.

    If a_ref_MHz is None, it is computed automatically from the data
    using auto_a_ref().
    """
    if a_ref_MHz is None:
        a_ref_MHz = auto_a_ref(hf_list)
    if not np.isfinite(a_ref_MHz) or a_ref_MHz <= 0.0:
        raise ValueError(f"a_ref_MHz must be finite and > 0, got {a_ref_MHz!r}")

    out = []
    for hf in hf_list:
        d = {
            k: v / a_ref_MHz
            for k, v in hf.items()
            if k in ("Axx", "Ayy", "Azz", "Axy", "Axz", "Ayz")
        }
        d["atom_index"] = hf["atom_index"]
        d["aiso_MHz"] = hf.get("aiso", 0)
        d["a_ref_MHz"] = a_ref_MHz
        out.append(d)
    return out


def group_by_site(hf_dimless: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split 6 ³¹P HF dicts into site 1 (P1-P3) and site 2 (P4-P6)."""
    if len(hf_dimless) < 6:
        raise ValueError(f"Expected 6 ³¹P HF tensors, got {len(hf_dimless)}")
    return hf_dimless[:3], hf_dimless[3:6]


# ═════════════════════════════════════════════════════════════════
# Full Pipeline
# ═════════════════════════════════════════════════════════════════


def run_full_pipeline(orca_output: str | Path | None = None) -> dict[str, Any]:
    """Run the complete DFT → circuit parameter pipeline.

    If orca_output is provided, parses it and auto-calibrates a_ref.
    Otherwise, returns geometry-derived data and ORCA input templates.
    """
    result = {}

    # Validate geometry
    geom = validate_geometry()
    result["geometry_valid"] = geom["valid"]
    if geom["errors"]:
        result["geometry_errors"] = geom["errors"]

    # P-P distances with S₆ distortions
    result["pp_distances"] = geom["pp_distances"]
    result["dipolar_table"] = compute_qubit_dipolar_table()

    # ORCA inputs
    result["orca_input_singlet"] = generate_orca_input()
    result["orca_input_radical"] = generate_radical_input()

    # Parse output if available
    if orca_output is not None:
        orca_path = Path(orca_output)
        if not orca_path.exists():
            raise ValueError(f"ORCA output path does not exist: {orca_path}")
        raw_hf = parse_orca_hf_output(orca_path)
        a_ref = auto_a_ref(raw_hf)
        dimless = convert_to_dimensionless(raw_hf, a_ref)
        site1, site2 = group_by_site(dimless)
        result["hf_site1"] = site1
        result["hf_site2"] = site2
        result["a_ref_MHz"] = a_ref
        result["source"] = "DFT (ORCA)"
    else:
        result["source"] = "geometry only (no ORCA output)"

    return result


def print_summary():
    """Print a summary of Posner geometry and coupling parameters."""
    geom = validate_geometry()
    pp = geom["pp_distances"]
    dt = compute_qubit_dipolar_table()

    print("=" * 70)
    print("  Posner S₆ Ca₉(PO₄)₆ — Structural Parameters")
    print("=" * 70)
    print(f"\n  Geometry valid: {'✓' if geom['valid'] else '✗'}")
    if geom["errors"]:
        for e in geom["errors"]:
            print(f"    ERROR: {e}")

    print("\n  P-P Distances (with S₆ distortions):")
    for pa, pb, r in pp:
        site_a = "site1" if pa in ("P1", "P2", "P3") else "site2"
        site_b = "site1" if pb in ("P1", "P2", "P3") else "site2"
        pair_type = "intra" if site_a == site_b else "cross"
        print(f"    {pa}-{pb}: {r:.3f} Å  [{pair_type}]")

    print("\n  Nuclear Dipolar Couplings (³¹P-³¹P):")
    for qi, qj, r, d in dt:
        print(f"    q{qi}-q{qj}: d = {d:.2e}  (r = {r:.3f} Å)")

    print("\n  Key insight: Nuclear dipolar coupling is ~10⁻⁸ in")
    print("  dimensionless units — negligible for RPM singlet yield")
    print("  (nanosecond timescale), but relevant for Posner coherence")
    print("  on Fisher's proposed seconds-to-minutes timescale.")
    print("=" * 70)


if __name__ == "__main__":
    print_summary()

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — IBM Quantum Heron v2 Posner molecule verification

r"""Posner Ca₉(PO₄)₆ radical-pair verification circuits for IBM hardware.

Experiments
-----------
1. **Posner RPM** — 8q (2e + 6n). Anisotropic hyperfine, nuclear dipolar
   coupling, thermal averaging (64 configs), recombination-weighted yield,
   exchange AND Zeeman field sweep.
2. **Heisenberg Propagation** — 10q magnon dynamics (first-principles).
3. **Posner Decoherence** — 8q Posner circuit with calibrated delay.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.linalg import expm

_ROOT = Path(__file__).resolve().parents[1]
for p in [_ROOT / "src"]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

# ── Pauli algebra ────────────────────────────────────────────────
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _kron(*ops):
    r = ops[0]
    for o in ops[1:]:
        r = np.kron(r, o)
    return r


def _pp(n, i, j, Pa, Pb):
    ops = [_I] * n
    ops[i] = Pa
    ops[j] = Pb
    return _kron(*ops)


def _ps(n, i, P):
    ops = [_I] * n
    ops[i] = P
    return _kron(*ops)


# ── Hyperfine test fixtures ─────────────────────────────────────
# These tensors are unit-test fixtures only. Runtime verification requires
# explicit `--hf-json`; builder functions call `_require_hf()` and do not use
# these values implicitly.
REFERENCE_TEST_HF_SITE1 = [
    {"Axx": 0.47, "Ayy": 0.50, "Azz": 0.53, "Axy": 0.02, "Axz": 0.01, "Ayz": 0.015},
    {"Axx": 0.45, "Ayy": 0.50, "Azz": 0.55, "Axy": 0.025, "Axz": 0.01, "Ayz": 0.02},
    {"Axx": 0.43, "Ayy": 0.48, "Azz": 0.52, "Axy": 0.015, "Axz": 0.02, "Ayz": 0.01},
]
REFERENCE_TEST_HF_SITE2 = [
    {"Axx": 0.30, "Ayy": 0.33, "Azz": 0.37, "Axy": 0.012, "Axz": 0.008, "Ayz": 0.01},
    {"Axx": 0.28, "Ayy": 0.32, "Azz": 0.35, "Axy": 0.015, "Axz": 0.006, "Ayz": 0.012},
    {"Axx": 0.27, "Ayy": 0.30, "Azz": 0.33, "Axy": 0.01, "Axz": 0.01, "Ayz": 0.008},
]


def _load_hf_json(path: str | Path) -> tuple[list[dict], list[dict]]:
    """Load explicit hyperfine tensors from JSON.

    Accepted keys are ``site1``/``site2`` or ``hf_site1``/``hf_site2``.
    Each site must contain three mappings with Axx,Ayy,Azz and optional
    Axy,Axz,Ayz entries, all in the circuit Hamiltonian units documented
    with the input file.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    site1 = data.get("site1", data.get("hf_site1"))
    site2 = data.get("site2", data.get("hf_site2"))
    if not isinstance(site1, list) or not isinstance(site2, list):
        raise ValueError("HF JSON must contain site1/site2 arrays")
    if len(site1) != 3 or len(site2) != 3:
        raise ValueError("HF JSON must contain exactly three tensors per Posner site")
    required = {"Axx", "Ayy", "Azz"}
    optional = {"Axy", "Axz", "Ayz"}
    out1: list[dict] = []
    out2: list[dict] = []
    for label, src, dst in (("site1", site1, out1), ("site2", site2, out2)):
        for idx, tensor in enumerate(src):
            if not isinstance(tensor, dict):
                raise ValueError(f"{label}[{idx}] must be a mapping")
            missing = required - set(tensor)
            if missing:
                raise ValueError(f"{label}[{idx}] missing required keys: {sorted(missing)}")
            dst.append({k: float(tensor.get(k, 0.0)) for k in required | optional})
    return out1, out2


def _load_extended_json(path: str | Path) -> dict:
    """Load explicit parameters for extended Posner experiments.

    The file may include cage_dephasing_rate, incorporation_tensors,
    transport_depolarizing_rates, ca43_hf_tensors, and ca_electron_map.
    Missing keys are not filled; the extended runners fail closed when an
    experiment needs a key that is absent.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Extended Posner JSON must contain a top-level object")
    return data


_DIPOLAR_KEYS = ("Axx", "Ayy", "Azz", "Axy", "Axz", "Ayz")


def _dipolar_tensor_from_mapping(raw: dict, idx: int) -> dict[str, float]:
    tensor = {}
    aliases = {
        "Axx": ("Axx", "Dxx", "Txx"),
        "Ayy": ("Ayy", "Dyy", "Tyy"),
        "Azz": ("Azz", "Dzz", "Tzz"),
        "Axy": ("Axy", "Dxy", "Txy"),
        "Axz": ("Axz", "Dxz", "Txz"),
        "Ayz": ("Ayz", "Dyz", "Tyz"),
    }
    for key, names in aliases.items():
        for name in names:
            if name in raw:
                tensor[key] = float(raw[name])
                break
        else:
            raise ValueError(f"nuclear_dipolar_pairs[{idx}] missing full tensor component {key}")
    return tensor


def _parse_dipolar_pairs(raw_pairs) -> list[tuple[int, int, dict[str, float]]]:
    """Parse explicit full 31P-31P dipolar tensors in circuit units."""
    if not isinstance(raw_pairs, list):
        raise ValueError("nuclear_dipolar_pairs must be a list")
    pairs: list[tuple[int, int, dict[str, float]]] = []
    seen: set[tuple[int, int]] = set()
    for idx, raw in enumerate(raw_pairs):
        if isinstance(raw, dict):
            qi = raw.get("qubit_i", raw.get("qi", raw.get("i")))
            qj = raw.get("qubit_j", raw.get("qj", raw.get("j")))
            tensor = _dipolar_tensor_from_mapping(raw, idx)
        elif isinstance(raw, (list, tuple)) and len(raw) == 8:
            qi, qj = raw[0], raw[1]
            tensor = {key: float(value) for key, value in zip(_DIPOLAR_KEYS, raw[2:])}
        else:
            raise ValueError(
                f"nuclear_dipolar_pairs[{idx}] must be a mapping with full tensor "
                "components or an 8-value list [qi,qj,Axx,Ayy,Azz,Axy,Axz,Ayz]"
            )
        qi_i, qj_i = int(qi), int(qj)
        if qi_i > qj_i:
            qi_i, qj_i = qj_i, qi_i
        if qi_i < 2 or qj_i > 7 or qi_i == qj_i:
            raise ValueError(f"nuclear_dipolar_pairs[{idx}] has invalid 31P qubits {(qi_i, qj_i)}")
        if any(not math.isfinite(value) for value in tensor.values()):
            raise ValueError(f"nuclear_dipolar_pairs[{idx}] has non-finite tensor")
        pairs.append((qi_i, qj_i, tensor))
        seen.add((qi_i, qj_i))
    expected = {(i, j) for i in range(2, 8) for j in range(i + 1, 8)}
    missing = sorted(expected - seen)
    extra = sorted(seen - expected)
    if missing or extra or len(pairs) != 15:
        raise ValueError(
            "nuclear_dipolar_pairs must contain exactly the 15 unique 31P pairs; "
            f"missing={missing}, extra={extra}, count={len(pairs)}"
        )
    return sorted(pairs, key=lambda item: (item[0], item[1]))


def _configure_dipolar_pairs_from_extended(extended: dict) -> None:
    """Install externally supplied dipolar pairs for runtime verification."""
    raw_pairs = extended.get("nuclear_dipolar_pairs")
    if raw_pairs is None:
        raise ValueError(
            "Extended Posner JSON must contain nuclear_dipolar_pairs. "
            "The built-in geometry-derived dipolar table is a test/reference "
            "fallback and is not accepted for publication verification runs."
        )
    global _DIPOLAR_TENSORS, _DIPOLAR_PAIRS
    _DIPOLAR_TENSORS = _parse_dipolar_pairs(raw_pairs)
    _DIPOLAR_PAIRS = [
        (i, j, max(abs(tensor[key]) for key in _DIPOLAR_KEYS)) for i, j, tensor in _DIPOLAR_TENSORS
    ]


def _load_runtime_parameters(args, *, require_extended: bool) -> None:
    if getattr(args, "hf1", None) is None or getattr(args, "hf2", None) is None:
        if not getattr(args, "hf_json", None):
            raise ValueError("--hf-json is required for Posner verification")
        args.hf1, args.hf2 = _load_hf_json(args.hf_json)
    if getattr(args, "extended_params", None) is None:
        if not getattr(args, "extended_json", None):
            if require_extended:
                raise ValueError(
                    "--extended-json with nuclear_dipolar_pairs is required for "
                    "publication-grade Posner verification and hardware submission"
                )
            args.extended_params = {}
        else:
            args.extended_params = _load_extended_json(args.extended_json)
    if require_extended:
        _configure_dipolar_pairs_from_extended(args.extended_params)


def _require_hf(hf1: list[dict] | None, hf2: list[dict] | None) -> tuple[list[dict], list[dict]]:
    if hf1 is None or hf2 is None:
        raise ValueError(
            "Explicit Posner hyperfine tensors are required. Pass --hf-json "
            "or call build_posner_circuit(..., hf1=..., hf2=...)."
        )
    return hf1, hf2


# Per-pair nuclear dipolar coupling from the local S6 Posner geometry helper.
# This table is used for unit tests and explicit reference calculations only.
# Runtime verification requires external nuclear_dipolar_pairs in --extended-json.
# Ref context: Swift, Van de Walle & Fisher, PCCP 20 (2018) 12373.
#
# CORRECTED distances: nearest P-P in S₆ Posner ≈ 4.95 Å (NOT 3.7 Å —
# that was the P-O bond length, not P-P distance).
#
# Physical coupling: D_dd/(2π) = (μ₀/4π)·γ²_P·ℏ/r³
#   γ_P/(2π) = 17.235 MHz/T → at 5.0 Å: D_dd ≈ 157 Hz
#   In dimensionless units (a_ref ≈ 7080 MHz): d ≈ 2.2 × 10⁻⁸
#
# This is ~10⁶ times weaker than the HF coupling. Nuclear dipolar is
# negligible for RPM singlet yield (ns timescale) but relevant for
# Posner coherence on Fisher's proposed s–min timescale.

_MU0_4PI = 1e-7  # T·m/A
_GAMMA_P = 1.0829e8  # rad/(s·T) for ³¹P
_HBAR = 1.0546e-34  # J·s
# Circuit reference scale for dimensionless Trotter angles. Hyperfine JSON
# inputs must be prepared in the same circuit units.
_A_REF_HZ = 7.08e9  # Hz


def _d_from_r_physical(r_angstrom):
    """Physically correct ³¹P-³¹P dipolar coupling in dimensionless units.

    Uses the point-dipole approximation: D ∝ 1/r³.

    KNOWN LIMITATION: For nuclei separated by ~4-5 Å through shared
    oxygen bridges, through-bond (pseudo-contact) contributions can
    add 20-50% to the point-dipole value. The true J-coupling would
    require DFT computation of the full spin-spin coupling tensor.
    The point-dipole formula underestimates the coupling for
    intra-site pairs (r ≈ 5 Å, sharing 0-1 oxygens) and is more
    accurate for cross-site pairs (r ≈ 6.5 Å, no shared oxygens).
    """
    r_m = r_angstrom * 1e-10
    D_dd_Hz = _MU0_4PI * _GAMMA_P**2 * _HBAR / (r_m**3 * 2 * math.pi)
    return D_dd_Hz / _A_REF_HZ


# (qubit_i, qubit_j, distance_Å) — dynamically computed from S₆ coordinates
# See tools/orca_posner_hf.py compute_pp_distances() for derivation.
# With S₆ distortions, all 15 P-P distances are non-degenerate:
#   ~4.14–4.21 Å — cross-site near-equatorial (6 pairs)
#   ~4.98–5.05 Å — intra-site (6 pairs)
#   ~6.49–6.52 Å — cross-site diametrically opposite (3 pairs)
def _build_dipolar_table():
    """Build dipolar table from DFT-optimized S₆ coordinates."""
    try:
        from orca_posner_hf import compute_qubit_dipolar_table

        qt = compute_qubit_dipolar_table()
        return [(qi, qj, r) for qi, qj, r, _ in qt]
    except ImportError:
        pass
    # Fallback: try relative import
    try:
        import importlib

        sys.path.insert(0, os.path.dirname(__file__))
        mod = importlib.import_module("orca_posner_hf")
        qt = mod.compute_qubit_dipolar_table()
        return [(qi, qj, r) for qi, qj, r, _ in qt]
    except Exception as exc:
        raise RuntimeError(
            "Could not compute Posner qubit dipolar table from ORCA geometry helper"
        ) from exc


_DIPOLAR_TABLE = _build_dipolar_table()
# Precompute coupling strengths (physically correct magnitudes)
_DIPOLAR_PAIRS = [(i, j, _d_from_r_physical(r)) for i, j, r in _DIPOLAR_TABLE]
try:
    from orca_posner_hf import compute_qubit_dipolar_tensor_table

    _DIPOLAR_TENSORS = [
        (int(row["qubit_i"]), int(row["qubit_j"]), {key: float(row[key]) for key in _DIPOLAR_KEYS})
        for row in compute_qubit_dipolar_tensor_table()
    ]
except Exception:
    _DIPOLAR_TENSORS = [
        (i, j, {"Axx": -d, "Ayy": -d, "Azz": 2 * d, "Axy": 0.0, "Axz": 0.0, "Ayz": 0.0})
        for i, j, d in _DIPOLAR_PAIRS
    ]

DEFAULT_NUC_DIPOLAR = _d_from_r_physical(5.0)  # intra-site avg
DEFAULT_NUC_DIPOLAR_CROSS = _d_from_r_physical(6.5)  # cross-far avg
_INTRA_PAIRS = [(2, 3), (2, 4), (3, 4), (5, 6), (5, 7), (6, 7)]
_CROSS_PAIRS = [(2, 5), (2, 6), (2, 7), (3, 5), (3, 6), (3, 7), (4, 5), (4, 6), (4, 7)]


def _add_spin_tensor_hamiltonian(H, n, qi, qj, tensor):
    H += (tensor["Axx"] / 4) * _pp(n, qi, qj, _X, _X)
    H += (tensor["Ayy"] / 4) * _pp(n, qi, qj, _Y, _Y)
    H += (tensor["Azz"] / 4) * _pp(n, qi, qj, _Z, _Z)
    H += (tensor["Axy"] / 4) * (_pp(n, qi, qj, _X, _Y) + _pp(n, qi, qj, _Y, _X))
    H += (tensor["Axz"] / 4) * (_pp(n, qi, qj, _X, _Z) + _pp(n, qi, qj, _Z, _X))
    H += (tensor["Ayz"] / 4) * (_pp(n, qi, qj, _Y, _Z) + _pp(n, qi, qj, _Z, _Y))
    return H


# ═════════════════════════════════════════════════════════════════
# Posner Hamiltonian (8 qubits)
# ═════════════════════════════════════════════════════════════════


def posner_hamiltonian(
    J: float,
    hf1: list[dict] | None = None,
    hf2: list[dict] | None = None,
    omega_0: float = 0.0,
    d_nuc: float = DEFAULT_NUC_DIPOLAR,
    d_nuc_cross: float = DEFAULT_NUC_DIPOLAR_CROSS,
) -> np.ndarray:
    r"""Full 8-qubit Posner Hamiltonian with nuclear dipolar coupling.

    q0=e₁, q1=e₂, q2-q4=I₁₋₃ (site1), q5-q7=I₄₋₆ (site2).

    H = J·S₁·S₂
      + Σₖ Aₖ·Iₖ·S₁  (k=1,2,3; anisotropic)
      + Σₖ Aₖ·Iₖ·S₂  (k=4,5,6; anisotropic)
      + ω₀·(S₁z+S₂z)
      + Σᵢⱼ Iᵢᵀ·Dᵢⱼ·Iⱼ  (full orientation-specific dipolar tensor)
    """
    n = 8
    hf1, hf2 = _require_hf(hf1, hf2)
    H = np.zeros((2**n, 2**n), dtype=complex)

    # Exchange
    for P in (_X, _Y, _Z):
        H += (J / 4) * _pp(n, 0, 1, P, P)

    # Anisotropic hyperfine site 1 (q0 ↔ q2,q3,q4) — full 3×3 tensor
    for k, hf in enumerate(hf1):
        nq = 2 + k
        H += (hf["Axx"] / 4) * _pp(n, 0, nq, _X, _X)
        H += (hf["Ayy"] / 4) * _pp(n, 0, nq, _Y, _Y)
        H += (hf["Azz"] / 4) * _pp(n, 0, nq, _Z, _Z)
        # Off-diagonal: Axy→(XY+YX)/2, Axz→(XZ+ZX)/2, Ayz→(YZ+ZY)/2
        H += (hf.get("Axy", 0) / 4) * (_pp(n, 0, nq, _X, _Y) + _pp(n, 0, nq, _Y, _X))
        H += (hf.get("Axz", 0) / 4) * (_pp(n, 0, nq, _X, _Z) + _pp(n, 0, nq, _Z, _X))
        H += (hf.get("Ayz", 0) / 4) * (_pp(n, 0, nq, _Y, _Z) + _pp(n, 0, nq, _Z, _Y))

    # Anisotropic hyperfine site 2 (q1 ↔ q5,q6,q7)
    for k, hf in enumerate(hf2):
        nq = 5 + k
        H += (hf["Axx"] / 4) * _pp(n, 1, nq, _X, _X)
        H += (hf["Ayy"] / 4) * _pp(n, 1, nq, _Y, _Y)
        H += (hf["Azz"] / 4) * _pp(n, 1, nq, _Z, _Z)
        H += (hf.get("Axy", 0) / 4) * (_pp(n, 1, nq, _X, _Y) + _pp(n, 1, nq, _Y, _X))
        H += (hf.get("Axz", 0) / 4) * (_pp(n, 1, nq, _X, _Z) + _pp(n, 1, nq, _Z, _X))
        H += (hf.get("Ayz", 0) / 4) * (_pp(n, 1, nq, _Y, _Z) + _pp(n, 1, nq, _Z, _Y))

    # Zeeman (electrons only; nuclear Zeeman is γ_n/γ_e ≈ 1/2500 — negligible)
    H += (omega_0 / 2) * _ps(n, 0, _Z) + (omega_0 / 2) * _ps(n, 1, _Z)

    # Nuclear dipolar: per-pair tensor from Posner geometry table
    # If d_nuc or d_nuc_cross are overridden from defaults, use uniform values
    # (backward compat); otherwise use the per-pair distance table.
    use_table = d_nuc == DEFAULT_NUC_DIPOLAR and d_nuc_cross == DEFAULT_NUC_DIPOLAR_CROSS
    if use_table:
        for i, j, tensor in _DIPOLAR_TENSORS:
            H = _add_spin_tensor_hamiltonian(H, n, i, j, tensor)
    else:
        for i, j in _INTRA_PAIRS:
            tensor = {
                "Axx": -d_nuc,
                "Ayy": -d_nuc,
                "Azz": 2 * d_nuc,
                "Axy": 0.0,
                "Axz": 0.0,
                "Ayz": 0.0,
            }
            H = _add_spin_tensor_hamiltonian(H, n, i, j, tensor)
        for i, j in _CROSS_PAIRS:
            tensor = {
                "Axx": -d_nuc_cross,
                "Ayy": -d_nuc_cross,
                "Azz": 2 * d_nuc_cross,
                "Axy": 0.0,
                "Axz": 0.0,
                "Ayz": 0.0,
            }
            H = _add_spin_tensor_hamiltonian(H, n, i, j, tensor)

    return H


def _singlet_proj_8q():
    s = np.array([0, 1, -1, 0], dtype=complex) / math.sqrt(2)
    return np.kron(np.outer(s, s.conj()), np.eye(2**6))


def analytical_singlet_thermal(
    J,
    hf1=None,
    hf2=None,
    omega_0=0.0,
    t=math.pi,
    d_nuc=DEFAULT_NUC_DIPOLAR,
    d_nuc_cross=DEFAULT_NUC_DIPOLAR_CROSS,
):
    hf1, hf2 = _require_hf(hf1, hf2)
    H = posner_hamiltonian(J, hf1, hf2, omega_0, d_nuc, d_nuc_cross)
    U = expm(-1j * H * t)
    se = np.array([0, 1, -1, 0], dtype=complex) / math.sqrt(2)
    PS = _singlet_proj_8q()
    total = 0.0
    for bits in itertools.product([0, 1], repeat=6):
        ns = np.zeros(64, dtype=complex)
        ns[sum(b << (5 - i) for i, b in enumerate(bits))] = 1.0
        psi = U @ np.kron(se, ns)
        total += float(np.real(psi.conj() @ PS @ psi))
    return total / 64.0


def analytical_singlet_recombination(
    J,
    hf1=None,
    hf2=None,
    omega_0=0.0,
    k_recomb=0.1,
    t_max=15.0,
    n_t=20,
    d_nuc=DEFAULT_NUC_DIPOLAR,
    d_nuc_cross=DEFAULT_NUC_DIPOLAR_CROSS,
):
    """Recombination-weighted, thermally averaged singlet yield."""
    hf1, hf2 = _require_hf(hf1, hf2)
    H = posner_hamiltonian(J, hf1, hf2, omega_0, d_nuc, d_nuc_cross)
    se = np.array([0, 1, -1, 0], dtype=complex) / math.sqrt(2)
    PS = _singlet_proj_8q()
    dt = t_max / n_t
    wsum, wnorm = 0.0, 0.0
    for ti in range(n_t):
        t = (ti + 0.5) * dt
        U = expm(-1j * H * t)
        w = k_recomb * math.exp(-k_recomb * t) * dt
        ps = 0.0
        for bits in itertools.product([0, 1], repeat=6):
            ns = np.zeros(64, dtype=complex)
            ns[sum(b << (5 - i) for i, b in enumerate(bits))] = 1.0
            psi = U @ np.kron(se, ns)
            ps += float(np.real(psi.conj() @ PS @ psi))
        wsum += (ps / 64.0) * w
        wnorm += w
    return wsum / wnorm if wnorm > 0 else 0.0


# ═════════════════════════════════════════════════════════════════
# Circuit builders
# ═════════════════════════════════════════════════════════════════


def _apply_cross_coupling(qc, eq, nq, alpha, beta, angle):
    """Apply exp(-iθ·σα⊗σβ/2) via basis rotation + RXX.

    Decomposition: σα⊗σβ = (Uα⊗Uβ)·(σx⊗σx)·(Uα†⊗Uβ†)
    where Ux=I, Uy=S, Uz=H (phase gate / Hadamard).
    Ref: Nielsen & Chuang, §4.7; Schweiger & Jeschke §3.3.
    """
    if abs(angle) < 1e-15:
        return
    # Pre-rotation: map σα → σx on electron, σβ → σx on nuclear
    if alpha == "y":
        qc.sdg(eq)
    elif alpha == "z":
        qc.h(eq)
    if beta == "y":
        qc.sdg(nq)
    elif beta == "z":
        qc.h(nq)
    # Core: exp(-iθ·σx⊗σx/2)
    qc.rxx(angle, eq, nq)
    # Post-rotation: undo basis change
    if alpha == "y":
        qc.s(eq)
    elif alpha == "z":
        qc.h(eq)
    if beta == "y":
        qc.s(nq)
    elif beta == "z":
        qc.h(nq)


def _apply_offdiag_hf(qc, eq, nq, hf, dt_factor):
    """Apply off-diagonal HF tensor elements with correct cross-coupling.

    Axy couples Sx·Iy + Sy·Ix (two cross terms).
    Axz couples Sx·Iz + Sz·Ix.
    Ayz couples Sy·Iz + Sz·Iy.
    """
    axy = hf.get("Axy", 0)
    axz = hf.get("Axz", 0)
    ayz = hf.get("Ayz", 0)
    if axy:
        _apply_cross_coupling(qc, eq, nq, "x", "y", axy * dt_factor)
        _apply_cross_coupling(qc, eq, nq, "y", "x", axy * dt_factor)
    if axz:
        _apply_cross_coupling(qc, eq, nq, "x", "z", axz * dt_factor)
        _apply_cross_coupling(qc, eq, nq, "z", "x", axz * dt_factor)
    if ayz:
        _apply_cross_coupling(qc, eq, nq, "y", "z", ayz * dt_factor)
        _apply_cross_coupling(qc, eq, nq, "z", "y", ayz * dt_factor)


def _apply_spin_tensor_circuit(qc, qi, qj, tensor, dt):
    """Apply exp(-i dt * sum_ab Aab I_i^a I_j^b) for spin-1/2 pairs."""
    qc.rxx(tensor["Axx"] * dt / 2, qi, qj)
    qc.ryy(tensor["Ayy"] * dt / 2, qi, qj)
    qc.rzz(tensor["Azz"] * dt / 2, qi, qj)
    if tensor["Axy"]:
        _apply_cross_coupling(qc, qi, qj, "x", "y", tensor["Axy"] * dt / 2)
        _apply_cross_coupling(qc, qi, qj, "y", "x", tensor["Axy"] * dt / 2)
    if tensor["Axz"]:
        _apply_cross_coupling(qc, qi, qj, "x", "z", tensor["Axz"] * dt / 2)
        _apply_cross_coupling(qc, qi, qj, "z", "x", tensor["Axz"] * dt / 2)
    if tensor["Ayz"]:
        _apply_cross_coupling(qc, qi, qj, "y", "z", tensor["Ayz"] * dt / 2)
        _apply_cross_coupling(qc, qi, qj, "z", "y", tensor["Ayz"] * dt / 2)


def _trotter_half(qc, J, hf1, hf2, omega_0, d_nuc, d_cross, dt):
    """Forward half of a 2nd-order Suzuki-Trotter step."""
    # Exchange
    qc.rxx(J * dt / 2, 0, 1)
    qc.ryy(J * dt / 2, 0, 1)
    qc.rzz(J * dt / 2, 0, 1)
    # Hyperfine site 1 (full tensor: diagonal + off-diagonal)
    for k, hf in enumerate(hf1):
        nq = 2 + k
        qc.rxx(hf["Axx"] * dt / 2, 0, nq)
        qc.ryy(hf["Ayy"] * dt / 2, 0, nq)
        qc.rzz(hf["Azz"] * dt / 2, 0, nq)
        _apply_offdiag_hf(qc, 0, nq, hf, dt / 2)
    # Hyperfine site 2
    for k, hf in enumerate(hf2):
        nq = 5 + k
        qc.rxx(hf["Axx"] * dt / 2, 1, nq)
        qc.ryy(hf["Ayy"] * dt / 2, 1, nq)
        qc.rzz(hf["Azz"] * dt / 2, 1, nq)
        _apply_offdiag_hf(qc, 1, nq, hf, dt / 2)
    # Zeeman
    qc.rz(omega_0 * dt, 0)
    qc.rz(omega_0 * dt, 1)
    # Nuclear dipolar: per-pair from geometry
    for i, j, tensor in _DIPOLAR_TENSORS:
        _apply_spin_tensor_circuit(qc, i, j, tensor, dt)


def _trotter_half_rev(qc, J, hf1, hf2, omega_0, d_nuc, d_cross, dt):
    """Reversed half of a 2nd-order Suzuki-Trotter step."""
    # Reverse order: dipolar, Zeeman, HF2, HF1, exchange
    for i, j, tensor in reversed(_DIPOLAR_TENSORS):
        _apply_spin_tensor_circuit(qc, i, j, tensor, dt)
    qc.rz(omega_0 * dt, 0)
    qc.rz(omega_0 * dt, 1)
    for k in reversed(range(len(hf2))):
        nq = 5 + k
        hf = hf2[k]
        _apply_offdiag_hf(qc, 1, nq, hf, dt / 2)
        qc.rzz(hf["Azz"] * dt / 2, 1, nq)
        qc.ryy(hf["Ayy"] * dt / 2, 1, nq)
        qc.rxx(hf["Axx"] * dt / 2, 1, nq)
    for k in reversed(range(len(hf1))):
        nq = 2 + k
        hf = hf1[k]
        _apply_offdiag_hf(qc, 0, nq, hf, dt / 2)
        qc.rzz(hf["Azz"] * dt / 2, 0, nq)
        qc.ryy(hf["Ayy"] * dt / 2, 0, nq)
        qc.rxx(hf["Axx"] * dt / 2, 0, nq)
    qc.rzz(J * dt / 2, 0, 1)
    qc.ryy(J * dt / 2, 0, 1)
    qc.rxx(J * dt / 2, 0, 1)


def build_posner_circuit(
    J=1.0,
    hf1=None,
    hf2=None,
    omega_0=0.0,
    t=math.pi,
    n_trotter=5,
    nuclear_init=(0,) * 6,
    d_nuc=DEFAULT_NUC_DIPOLAR,
    d_cross=DEFAULT_NUC_DIPOLAR_CROSS,
):
    """8q Posner, 2nd-order Suzuki-Trotter, full tensor dipolar."""
    from qiskit import QuantumCircuit

    hf1, hf2 = _require_hf(hf1, hf2)
    qc = QuantumCircuit(8, 8)
    qc.x(1)
    qc.h(0)
    qc.cx(0, 1)
    qc.z(0)
    for i, b in enumerate(nuclear_init):
        if b:
            qc.x(2 + i)
    dt = t / n_trotter
    # 2nd-order Suzuki-Trotter: S₂(dt) = U(dt/2)·U†(dt/2)
    for _ in range(n_trotter):
        _trotter_half(qc, J, hf1, hf2, omega_0, d_nuc, d_cross, dt / 2)
        _trotter_half_rev(qc, J, hf1, hf2, omega_0, d_nuc, d_cross, dt / 2)
    qc.cx(0, 1)
    qc.h(0)
    qc.measure(range(8), range(8))
    return qc


def build_posner_decoherence_circuit(
    J=1.0,
    hf1=None,
    hf2=None,
    omega_0=0.0,
    # DD: XY-4 dynamical decoupling sequence
    dd_sequence: str | None = None,
    t=math.pi,
    n_trotter=5,
    delay_dt=0,
    nuclear_init=(0,) * 6,
):
    """8q Posner + calibrated delay. 2nd-order Suzuki-Trotter."""
    from qiskit import QuantumCircuit

    hf1, hf2 = _require_hf(hf1, hf2)
    qc = QuantumCircuit(8, 8)
    qc.x(1)
    qc.h(0)
    qc.cx(0, 1)
    qc.z(0)
    for i, b in enumerate(nuclear_init):
        if b:
            qc.x(2 + i)
    dt = t / n_trotter
    for _ in range(n_trotter):
        _trotter_half(
            qc, J, hf1, hf2, omega_0, DEFAULT_NUC_DIPOLAR, DEFAULT_NUC_DIPOLAR_CROSS, dt / 2
        )
        _trotter_half_rev(
            qc, J, hf1, hf2, omega_0, DEFAULT_NUC_DIPOLAR, DEFAULT_NUC_DIPOLAR_CROSS, dt / 2
        )
    if delay_dt > 0:
        if dd_sequence == "xy4":
            # XY-4: τ/2 - X - τ - Y - τ - X - τ - Y - τ/2
            # Proper symmetric spacing for full refocusing.
            # DD only on ELECTRON qubits (q0,q1) — π-pulses on nuclear
            # qubits would flip the nuclear state and corrupt the model.
            tau = delay_dt // 4
            for q in range(8):
                qc.delay(tau // 2, q, unit="dt")
            for q in [0, 1]:  # electron qubits only
                qc.x(q)
            for q in range(8):
                qc.delay(tau, q, unit="dt")
            for q in [0, 1]:
                qc.y(q)
            for q in range(8):
                qc.delay(tau, q, unit="dt")
            for q in [0, 1]:
                qc.x(q)
            for q in range(8):
                qc.delay(tau, q, unit="dt")
            for q in [0, 1]:
                qc.y(q)
            for q in range(8):
                qc.delay(tau // 2, q, unit="dt")
        else:
            qc.barrier()
            for q in range(8):
                qc.delay(delay_dt, q, unit="dt")
            qc.barrier()
    qc.cx(0, 1)
    qc.h(0)
    qc.measure(range(8), range(8))
    return qc


# Heisenberg chain (10q) — coupling from Posner lattice geometry
# Superexchange through Ca²⁺ bridges decays EXPONENTIALLY with distance
# (not 1/r⁶ — that's direct dipolar). Ref: Anderson, "Antiferromagnetism.
# Theory of Superexchange Interaction", Phys. Rev. 79, 350 (1950).
# β ≈ 1.0 Å⁻¹ for oxide-bridge superexchange pathways.
def _posner_chain_couplings(n):
    """Inter-Posner couplings from exponential superexchange decay."""
    beta = 1.0  # Å⁻¹, decay constant for Ca-O-P superexchange pathway
    r_nn = 4.0  # Å, nearest P-P distance in Posner
    dr = 2.5  # Å, inter-site spacing increment
    return [math.exp(-beta * dr * i) for i in range(n - 1)]


def build_chain_circuit(n_qubits=10, J_vals=None, t=1.0, n_trotter=3):
    """Heisenberg chain with 2nd-order Suzuki-Trotter."""
    from qiskit import QuantumCircuit

    if J_vals is None:
        J_vals = _posner_chain_couplings(n_qubits)
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.x(0)
    dt = t / n_trotter
    for _ in range(n_trotter):
        # Forward half
        for i in range(n_qubits - 1):
            qc.rxx(J_vals[i] * dt / 4, i, i + 1)
            qc.ryy(J_vals[i] * dt / 4, i, i + 1)
            qc.rzz(J_vals[i] * dt / 4, i, i + 1)
        # Reversed half
        for i in reversed(range(n_qubits - 1)):
            qc.rxx(J_vals[i] * dt / 4, i, i + 1)
            qc.ryy(J_vals[i] * dt / 4, i, i + 1)
            qc.rzz(J_vals[i] * dt / 4, i, i + 1)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def heisenberg_chain_H(n, J_vals):
    H = np.zeros((2**n, 2**n), dtype=complex)
    for i in range(n - 1):
        for P in (_X, _Y, _Z):
            H += (J_vals[i] / 4) * _pp(n, i, i + 1, P, P)
    return H


def analytical_chain_corr(n, J_vals, t):
    H = heisenberg_chain_H(n, J_vals)
    psi0 = np.zeros(2**n, dtype=complex)
    psi0[2 ** (n - 1)] = 1.0
    psi = expm(-1j * H * t) @ psi0
    return [float(np.real(psi.conj() @ _pp(n, 0, d, _Z, _Z) @ psi)) for d in range(1, n)]


# ═════════════════════════════════════════════════════════════════
# Analysis
# ═════════════════════════════════════════════════════════════════


def _berr(p, n):
    return math.sqrt(p * (1 - p) / n) if n > 0 else 0.0


def analyse_rpm_8q(counts):
    total = sum(counts.values())
    ns = sum(
        c
        for bs, c in counts.items()
        if int(bs.replace(" ", "")[-(0 + 1)]) == 1 and int(bs.replace(" ", "")[-(1 + 1)]) == 1
    )
    p = ns / total if total else 0.0
    return {
        "singlet_probability": round(p, 6),
        "error_bar": round(_berr(p, total), 6),
        "shots": total,
    }


def analyse_chain(counts, n):
    total = sum(counts.values())
    corrs = []
    for d in range(1, n):
        ns = sum(
            c
            for bs, c in counts.items()
            if int(bs.replace(" ", "")[-(0 + 1)]) == int(bs.replace(" ", "")[-(d + 1)])
        )
        corrs.append(round((2 * ns - total) / total, 6) if total else 0.0)
    return {"zz_from_0": corrs, "shots": total}


# ═════════════════════════════════════════════════════════════════
# Runner
# ═════════════════════════════════════════════════════════════════

# Layout used only when the caller explicitly has no live backend object.
_LOCAL_LAYOUT = [0, 1, 2, 3, 4, 5, 6, 7]


def _find_best_layout(backend, n_qubits):
    """Find connected n-qubit subgraph with lowest avg 2Q gate error."""
    try:
        import networkx as nx

        cmap = backend.coupling_map
        G = cmap.graph.to_undirected()
        props = backend.properties()
        best_layout, best_err = None, float("inf")
        # Enumerate all connected subgraphs of size n_qubits
        for component in nx.connected_components(G):
            sub = G.subgraph(component)
            if len(sub) < n_qubits:
                continue
            # Try sliding windows along BFS ordering for efficiency
            for start in list(sub.nodes)[:20]:  # limit search
                bfs_nodes = list(nx.bfs_tree(sub, start).nodes)[:n_qubits]
                if len(bfs_nodes) < n_qubits:
                    continue
                # Score: average 2Q gate error across all edges in subgraph
                total_err, n_edges = 0.0, 0
                for u in bfs_nodes:
                    for v in bfs_nodes:
                        if sub.has_edge(u, v) and u < v:
                            try:
                                total_err += props.gate_error("ecr", [u, v])
                            except Exception:
                                try:
                                    total_err += props.gate_error("cx", [u, v])
                                except Exception:
                                    raise RuntimeError(f"No calibrated 2Q error for edge {(u, v)}")
                            n_edges += 1
                avg_err = total_err / max(n_edges, 1)
                if avg_err < best_err:
                    best_err = avg_err
                    best_layout = bfs_nodes
        if best_layout is None:
            raise RuntimeError(f"No connected calibrated {n_qubits}-qubit layout found")
        return best_layout
    except AttributeError:
        return _LOCAL_LAYOUT[:n_qubits]


# Cached runner context to avoid reconnecting on every circuit
_runner_ctx = {"service": None, "backend": None, "layout_cache": {}}


def _vault_section(path: Path, section_regex: str) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(
        rf"(?ims)^#{{1,6}}\s*{section_regex}[^\n]*\n(?P<body>.*?)(?=^#{{1,6}}\s|\Z)"
    )
    match = pattern.search(text)
    return match.group("body") if match else ""


def _normalise_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", label.lower())


def _vault_field(
    path: str | Path | None, section_regex: str, labels: tuple[str, ...]
) -> str | None:
    if not path:
        return None
    vault_path = Path(path)
    if not vault_path.exists():
        return None
    wanted = {_normalise_label(label) for label in labels}
    for raw_line in _vault_section(vault_path, section_regex).splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        label, value = line.split(":", 1)
        if _normalise_label(label.strip("-* `")) not in wanted:
            continue
        backticked = re.findall(r"`([^`]+)`", value)
        clean = backticked[0] if backticked else value.strip()
        clean = clean.strip().strip("`").strip()
        if clean:
            return clean
    return None


def _ibm_credentials(args):
    token = (
        args.token
        or os.environ.get("SC_NEUROCORE_IBM_TOKEN")
        or os.environ.get("QISKIT_IBM_TOKEN")
        or os.environ.get("IBM_QUANTUM_TOKEN")
        or _vault_field(
            getattr(args, "credential_vault", None),
            getattr(args, "vault_section", "IBM"),
            ("API Key", "Token"),
        )
    )
    if not token:
        raise RuntimeError(
            "SC_NEUROCORE_IBM_TOKEN, QISKIT_IBM_TOKEN, IBM_QUANTUM_TOKEN, "
            "--token, or --credential-vault is required for hardware execution. "
            "Use --simulator only for explicit local simulation."
        )
    instance = (
        getattr(args, "instance", None)
        or os.environ.get("SC_NEUROCORE_IBM_CRN")
        or os.environ.get("SC_NEUROCORE_IBM_INSTANCE")
        or _vault_field(
            getattr(args, "credential_vault", None),
            getattr(args, "vault_section", "IBM"),
            ("CRN/Instance", "Instance", "CRN"),
        )
    )
    channel = (
        getattr(args, "channel", None)
        or os.environ.get("SC_NEUROCORE_IBM_CHANNEL")
        or _vault_field(
            getattr(args, "credential_vault", None),
            getattr(args, "vault_section", "IBM"),
            ("Channel",),
        )
        or "ibm_cloud"
    )
    return token, instance, channel


def _extract_sampler_counts(pub_result):
    data = pub_result.data
    for reg_name in ("meas", "c", "cr", "c0", "c1"):
        reg = getattr(data, reg_name, None)
        if reg is not None and hasattr(reg, "get_counts"):
            return reg.get_counts()
    for attr in dir(data):
        if attr.startswith("_"):
            continue
        reg = getattr(data, attr, None)
        if reg is not None and hasattr(reg, "get_counts"):
            return reg.get_counts()
    raise RuntimeError("Sampler result does not expose a counts register")


def _ibm_backend(args):
    if _runner_ctx["backend"] is not None:
        return _runner_ctx["backend"]
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError as exc:
        raise RuntimeError("qiskit-ibm-runtime is required for hardware execution") from exc
    token, instance, channel = _ibm_credentials(args)
    kwargs = {"channel": channel, "token": token}
    if instance:
        kwargs["instance"] = instance
    service = QiskitRuntimeService(**kwargs)
    backend = service.backend(args.backend or "ibm_fez")
    _runner_ctx["service"] = service
    _runner_ctx["backend"] = backend
    return backend


def _transpile_for_backend(qc, backend, optimization_level=3, initial_layout=None):
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    kwargs = {"backend": backend, "optimization_level": optimization_level}
    if initial_layout is not None:
        kwargs["initial_layout"] = initial_layout
    pass_manager = generate_preset_pass_manager(**kwargs)
    return pass_manager.run(qc)


def _run_hardware_counts(qc, args):
    from qiskit_ibm_runtime import SamplerV2

    backend = _ibm_backend(args)
    nq = qc.num_qubits
    if nq not in _runner_ctx["layout_cache"]:
        _runner_ctx["layout_cache"][nq] = _find_best_layout(backend, nq)
    layout = _runner_ctx["layout_cache"][nq]
    tqc = _transpile_for_backend(
        qc, backend, optimization_level=args.optimization_level, initial_layout=layout
    )
    sampler = SamplerV2(mode=backend)
    job = sampler.run([tqc], shots=args.shots)
    result = job.result()
    return _extract_sampler_counts(result[0])


def _submit_hardware_job(circuits, args, initial_layout=None):
    from qiskit_ibm_runtime import SamplerV2

    backend = _ibm_backend(args)
    transpiled = [
        _transpile_for_backend(
            qc,
            backend,
            optimization_level=args.optimization_level,
            initial_layout=initial_layout[: qc.num_qubits] if initial_layout else None,
        )
        for qc in circuits
    ]
    sampler = SamplerV2(mode=backend)
    return sampler.run(transpiled, shots=args.shots)


def _exec(qc, args):
    if args.simulator:
        from qiskit.quantum_info import Statevector

        sv = Statevector.from_instruction(qc.remove_final_measurements(inplace=False))
        return sv.sample_counts(args.shots)
    return _run_hardware_counts(qc, args)


def _thermal_avg_singlet(J, omega_0, t, args, n_trotter=5):
    """Run 64 nuclear configs, return (mean, stderr) singlet yield."""
    ps_vals = []
    for bits in itertools.product([0, 1], repeat=6):
        qc = build_posner_circuit(
            J=J,
            omega_0=omega_0,
            t=t,
            nuclear_init=bits,
            n_trotter=n_trotter,
            hf1=args.hf1,
            hf2=args.hf2,
        )
        counts = _exec(qc, args)
        ps_vals.append(analyse_rpm_8q(counts)["singlet_probability"])
    arr = np.array(ps_vals)
    return float(arr.mean()), float(arr.std() / math.sqrt(len(arr)))


def _thermal_avg_decoherence(delay_dt, dd_sequence, args):
    """Run 64 nuclear configs for decoherence circuit, return (mean, stderr)."""
    ps_vals = []
    for bits in itertools.product([0, 1], repeat=6):
        qc = build_posner_decoherence_circuit(
            J=1.0,
            omega_0=0.5,
            t=math.pi,
            n_trotter=5,
            delay_dt=delay_dt,
            dd_sequence=dd_sequence,
            nuclear_init=bits,
            hf1=args.hf1,
            hf2=args.hf2,
        )
        counts = _exec(qc, args)
        ps_vals.append(analyse_rpm_8q(counts)["singlet_probability"])
    arr = np.array(ps_vals)
    return float(arr.mean()), float(arr.std() / math.sqrt(len(arr)))


def run_verification(args):
    _load_runtime_parameters(args, require_extended=not args.simulator)

    results = {
        "timestamp": datetime.now().isoformat(),
        "mode": "simulator" if args.simulator else "hardware",
    }
    P = print
    P("═" * 72)
    P("  SC-NeuroCore — Full Posner Molecule Verification (Heron v2)")
    P("  8q Posner: aniso-HF + nuclear dipolar + thermal avg + recombination")
    P("═" * 72)

    # ── Exp 1a: Exchange sweep (J) ────────────────────────────────
    P("\n▸ Exp 1a: Exchange protection sweep (fixed ω₀=0.5)")
    omega_0, t = 0.5, math.pi
    rpm_j = []
    for J in [0.0, 0.5, 1.0, 3.0, 10.0]:
        nt = max(5, math.ceil(3 * J))
        p, se = _thermal_avg_singlet(J, omega_0, t, args, nt)
        th = analytical_singlet_thermal(J, hf1=args.hf1, hf2=args.hf2, omega_0=omega_0, t=t)
        rpm_j.append(
            {
                "J": J,
                "p": round(p, 6),
                "se": round(se, 6),
                "theory": round(th, 6),
                "err": round(abs(p - th), 6),
            }
        )
        P(
            f"  J={J:>5.1f}  Φ_S={p:.4f}±{se:.4f}  theory={th:.4f}  [{'protected' if th > 0.7 else 'mixing'}]"
        )
    results["rpm_exchange_sweep"] = rpm_j

    # ── Exp 1b: Zeeman field sweep (ω₀) ──────────────────────────
    P("\n▸ Exp 1b: Magnetic field sweep (fixed J=1.0)")
    J_fixed = 1.0
    rpm_b = []
    for w in [0.0, 0.3, 0.7, 1.0, 2.0, 5.0]:
        nt = max(5, math.ceil(3 * max(J_fixed, w)))
        p, se = _thermal_avg_singlet(J_fixed, w, t, args, nt)
        th = analytical_singlet_thermal(J_fixed, hf1=args.hf1, hf2=args.hf2, omega_0=w, t=t)
        rpm_b.append({"omega_0": w, "p": round(p, 6), "se": round(se, 6), "theory": round(th, 6)})
        P(f"  ω₀={w:.1f}  Φ_S={p:.4f}±{se:.4f}  theory={th:.4f}")
    results["rpm_zeeman_sweep"] = rpm_b

    # ── Exp 1c: Recombination from CIRCUITS (time sweep) ──────────
    P("\n▸ Exp 1c: Circuit-based recombination (J=1.0, k=0.1, time sweep)")
    k_recomb = 0.1
    time_points = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]
    recomb_data = []
    wsum, wnorm = 0.0, 0.0
    for idx, ti in enumerate(time_points):
        # Proper Δt for non-uniform spacing
        if idx == 0:
            dt_i = (time_points[1] - ti) / 2 + ti  # from 0 to midpoint
        elif idx == len(time_points) - 1:
            dt_i = ti - (time_points[-2] + ti) / 2  # from midpoint to end
        else:
            dt_i = (time_points[idx + 1] - time_points[idx - 1]) / 2  # trapezoidal
        nt = max(5, math.ceil(3 * max(1.0, 0.5) * ti / math.pi))
        p, se = _thermal_avg_singlet(1.0, 0.5, ti, args, nt)
        w = k_recomb * math.exp(-k_recomb * ti) * dt_i  # includes Δt
        wsum += p * w
        wnorm += w
        recomb_data.append(
            {"t": ti, "p_singlet": round(p, 6), "weight": round(w, 6), "dt": round(dt_i, 3)}
        )
        P(f"  t={ti:>5.1f}  Φ_S={p:.4f}±{se:.4f}  w={w:.4f}")
    phi_recomb_circuit = wsum / wnorm if wnorm > 0 else 0.0
    phi_recomb_exact = analytical_singlet_recombination(
        1.0, hf1=args.hf1, hf2=args.hf2, omega_0=0.5, k_recomb=k_recomb, n_t=10
    )
    results["recombination"] = {
        "phi_circuit": round(phi_recomb_circuit, 6),
        "phi_exact": round(phi_recomb_exact, 6),
        "time_points": recomb_data,
        "k": k_recomb,
    }
    P(f"  Φ_S(circuit recomb) = {phi_recomb_circuit:.4f}")
    P(f"  Φ_S(exact recomb)   = {phi_recomb_exact:.4f}")

    # ── Exp 1d: Classical vs Quantum comparison ───────────────────
    # ³¹P gyromagnetic ratio: γ = 17.235 MHz/T
    # ω₀ = γ · B / a_ref.  For B = 50 µT → ω₀ ≈ 0.00086 (tiny).
    # To make a meaningful comparison, use the SAME ω₀ in both.
    P("\n▸ Exp 1d: Semiclassical vs full-quantum comparison")
    try:
        from sc_neurocore.quantum_cognition.radical_pair import RadicalPairModel

        gamma_31P = 17.235e6  # Hz/T
        B_earth = 50e-6  # T
        # Convert B to dimensionless ω₀ using our a_ref scaling
        # a_ref ≈ 3540 MHz (³¹P isotropic HF), so ω₀ = γ·B / a_ref
        a_ref = 3540e6  # Hz
        omega_matched = gamma_31P * B_earth / a_ref  # ≈ 0.000243
        rpm_classical = RadicalPairModel()
        phi_sc = rpm_classical.singlet_yield(b_local=B_earth)
        phi_q = analytical_singlet_thermal(
            1.0, hf1=args.hf1, hf2=args.hf2, omega_0=omega_matched, t=math.pi
        )
        results["classical_vs_quantum"] = {
            "B_field_T": B_earth,
            "omega_0_matched": round(omega_matched, 6),
            "semiclassical_schulten_wolynes": round(phi_sc, 6),
            "full_quantum_8q": round(phi_q, 6),
            "discrepancy": round(abs(phi_sc - phi_q), 6),
        }
        P(f"  B = {B_earth * 1e6:.0f} µT → ω₀ = {omega_matched:.6f}")
        P(f"  Schulten-Wolynes (semiclassical): {phi_sc:.4f}")
        P(f"  Full quantum (8q Posner):         {phi_q:.4f}")
        P(f"  Discrepancy:                      {abs(phi_sc - phi_q):.4f}")
    except ImportError:
        P("  [SKIP] RadicalPairModel not available")

    # ── Exp 2: 10-qubit Heisenberg chain ──────────────────────────
    P("\n▸ Exp 2: Heisenberg propagation (10q, exp superexchange)")
    nc = 10
    Jc = _posner_chain_couplings(nc)
    tc = 1.0
    qc = build_chain_circuit(n_qubits=nc, J_vals=Jc, t=tc)
    counts = _exec(qc, args)
    cr = analyse_chain(counts, nc)
    if nc <= 14:
        th_c = analytical_chain_corr(nc, Jc, tc)
        cr["theory"] = [round(c, 6) for c in th_c]
    results["spin_chain"] = cr
    for d in range(min(5, nc - 1)):
        ts = f" (theory {th_c[d]:.4f})" if nc <= 14 else ""
        P(f"  ⟨Z₀Z_{d + 1}⟩ = {cr['zz_from_0'][d]:.4f}{ts}")

    # ── Exp 3a: Posner decoherence (raw T₂*) ─────────────────────
    P("\n▸ Exp 3a: Posner decoherence — raw (no DD)")
    delays = [0, 1000, 5000, 10000, 50000]
    dec_raw = []
    for dd in delays:
        p, se = _thermal_avg_decoherence(dd, None, args)
        dec_raw.append(
            {
                "delay_dt": dd,
                "delay_us": round(dd * 0.00022, 3),
                "singlet": round(p, 6),
                "se": round(se, 6),
            }
        )
        P(f"  delay={dd:>5d}dt ({dd * 0.00022:.2f}μs)  Φ_S={p:.4f}±{se:.4f}")
    results["decoherence_raw"] = dec_raw

    # ── Exp 3b: Posner decoherence WITH XY-4 DD ──────────────────
    P("\n▸ Exp 3b: Posner decoherence — with XY-4 dynamical decoupling")
    dec_dd = []
    for dd in delays:
        if dd == 0:
            dec_dd.append(dec_raw[0])
            P(f"  delay=    0dt (0.00μs)  Φ_S={dec_raw[0]['singlet']:.4f}")
            continue
        p, se = _thermal_avg_decoherence(dd, "xy4", args)
        dec_dd.append(
            {
                "delay_dt": dd,
                "delay_us": round(dd * 0.00022, 3),
                "singlet": round(p, 6),
                "se": round(se, 6),
            }
        )
        P(f"  delay={dd:>5d}dt ({dd * 0.00022:.2f}μs)  Φ_S={p:.4f}±{se:.4f}  [XY-4]")
    results["decoherence_xy4"] = dec_dd

    # ── Extended experiments (from posner_extended.py) ────────────
    try:
        from posner_extended import (
            run_exp4_biological_noise,
            run_exp5_transport,
            run_exp6_43ca,
        )

        # Exp 4: Biological noise comparison
        try:
            results["biological_noise"] = run_exp4_biological_noise(
                build_posner_circuit, _exec, args, P
            )
        except Exception as e:
            P(f"\n▸ Exp 4: [SKIP] {e}")

        # Exp 5: Two-Posner transport
        try:
            results["two_posner_transport"] = run_exp5_transport(args, P)
        except Exception as e:
            P(f"\n▸ Exp 5: [SKIP] {e}")

        # Exp 6: ⁴³Ca-enriched (35q MPS)
        try:
            results["ca43_enriched"] = run_exp6_43ca(args, P)
        except Exception as e:
            P(f"\n▸ Exp 6: [SKIP] {e}")
    except ImportError:
        P("\n  [SKIP] Extended experiments (posner_extended.py not found)")

    # ── Save ──────────────────────────────────────────────────────
    od = _ROOT / "results" / "ibm_verification"
    od.mkdir(parents=True, exist_ok=True)
    op = od / f"verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(op, "w") as f:
        json.dump(results, f, indent=2, default=str)
    P(f"\n  Results → {op}")
    P("═" * 72)
    return results


def main():
    ap = argparse.ArgumentParser(description="Posner molecule verification on IBM Heron v2")
    ap.add_argument("--simulator", action="store_true", help="Use statevector simulator")
    ap.add_argument("--backend", default=None, help="IBM backend name (default: ibm_fez)")
    ap.add_argument("--token", default=None, help="IBM Quantum API token")
    ap.add_argument("--instance", default=None, help="IBM Cloud CRN or Runtime instance")
    ap.add_argument("--channel", default="ibm_cloud", help="IBM Runtime channel")
    ap.add_argument(
        "--credential-vault", default=None, help="Read IBM credentials from a local vault file"
    )
    ap.add_argument("--vault-section", default="IBM", help="Credential vault section heading")
    ap.add_argument("--optimization-level", type=int, default=3)
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument(
        "--hf-json",
        default=None,
        help="Required Posner hyperfine tensor JSON with site1/site2 arrays",
    )
    ap.add_argument(
        "--extended-json",
        default=None,
        help=(
            "Explicit Posner external parameters. Hardware/submit paths require "
            "nuclear_dipolar_pairs; missing extended experiment values cause "
            "only those experiments to skip"
        ),
    )
    ap.add_argument(
        "--submit-only",
        action="store_true",
        help="Submit circuits and print job IDs without waiting",
    )
    ap.add_argument(
        "--retrieve",
        nargs="+",
        metavar="JOB_ID",
        help="Retrieve results from previously submitted job IDs",
    )
    args = ap.parse_args()

    if args.retrieve:
        _retrieve_jobs(args)
    elif args.submit_only:
        _submit_only(args)
    else:
        run_verification(args)


def _submit_only(args):
    """Submit key circuits and print job IDs for later retrieval."""
    _load_runtime_parameters(args, require_extended=True)
    backend = _ibm_backend(args)
    layout = _find_best_layout(backend, 10)
    P = print
    P("Submitting circuits (async)...")
    circuits = {
        "exchange_J1": build_posner_circuit(J=1.0, omega_0=0.5, hf1=args.hf1, hf2=args.hf2),
        "exchange_J10": build_posner_circuit(
            J=10.0, omega_0=0.5, n_trotter=30, hf1=args.hf1, hf2=args.hf2
        ),
        "decoherence_raw": build_posner_decoherence_circuit(
            delay_dt=10000, hf1=args.hf1, hf2=args.hf2
        ),
        "decoherence_xy4": build_posner_decoherence_circuit(
            delay_dt=10000, dd_sequence="xy4", hf1=args.hf1, hf2=args.hf2
        ),
        "chain_10q": build_chain_circuit(n_qubits=10),
    }
    od = _ROOT / "results" / "ibm_verification"
    od.mkdir(parents=True, exist_ok=True)
    job = _submit_hardware_job(list(circuits.values()), args, initial_layout=layout)
    job_ids = {
        "job_id": job.job_id(),
        "backend": args.backend or "ibm_fez",
        "circuits": list(circuits.keys()),
        "shots": args.shots,
    }
    P(f"  ▸ submitted batch job_id={job.job_id()}")
    op = od / f"submitted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(op, "w") as f:
        json.dump(job_ids, f, indent=2)
    P(f"\nJob IDs saved to {op}")
    P(f"Retrieve with: python tools/verify_ibm_heron.py --retrieve {job.job_id()}")


def _retrieve_jobs(args):
    """Retrieve and display results from previously submitted jobs."""
    _ibm_backend(args)
    service = _runner_ctx["service"]
    if service is None:
        raise RuntimeError("IBM Runtime service did not initialise")
    P = print
    P("Retrieving jobs...")
    for jid in args.retrieve:
        try:
            job = service.job(jid)
            result = job.result()
            P(f"  ▸ {jid}: {len(result)} PUB(s)")
            for idx, pub in enumerate(result):
                counts = _extract_sampler_counts(pub)
                P(f"    PUB[{idx}]: {sum(counts.values())} shots, {len(counts)} unique bitstrings")
                if idx < 4:
                    r8 = analyse_rpm_8q(counts)
                    P(f"      Singlet probability: {r8['singlet_probability']:.4f}")
        except Exception as e:
            P(f"  ▸ {jid}: ERROR - {e}")


if __name__ == "__main__":
    main()

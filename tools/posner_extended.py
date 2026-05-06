# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Extended Posner experiments (43Ca, bio noise, transport)

r"""Extended Posner molecule verification beyond the 8q ³¹P model.

Experiments:
4. **Biological noise comparison** — Same circuit under ideal, QPU, and
   calibrated 310K biological noise models.
5. **Two-Posner transport** — 16q entanglement swapping circuit modeling
   synthesis → diffusion → binding → Ca²⁺ release.
6. **⁴³Ca-enriched Posner** — 35q model with I=7/2 calcium nuclear spins
   using MPS simulator for reference.
"""

from __future__ import annotations
import math
from typing import Any
import numpy as np

_SPIN_TENSOR_KEYS = ("Axx", "Ayy", "Azz", "Axy", "Axz", "Ayz")
_INCORPORATION_PAIR_KEYS = ("p1_p2", "p1_p3", "p2_p3")
_CA43_START_QUBITS = (8, 11, 14, 17, 20, 23, 26, 29, 32)


def _require_spin_tensor(name: str, tensor: dict[str, float]) -> dict[str, float]:
    missing = [key for key in _SPIN_TENSOR_KEYS if key not in tensor]
    if missing:
        raise ValueError(f"{name} is missing tensor components: {', '.join(missing)}")
    return {key: float(tensor[key]) for key in _SPIN_TENSOR_KEYS}


def _require_three_hf(name: str, tensors: list[dict] | None) -> list[dict[str, float]]:
    if tensors is None:
        raise ValueError(f"{name} is required; no bundled hyperfine tensors are used")
    if len(tensors) != 3:
        raise ValueError(f"{name} must contain exactly 3 tensor dictionaries")
    return [
        _require_spin_tensor(f"{name}[{idx}]", dict(tensor)) for idx, tensor in enumerate(tensors)
    ]


def _require_incorporation_tensors(
    name: str,
    tensors: dict[str, dict[str, float]] | None,
) -> dict[str, dict[str, float]]:
    if tensors is None:
        raise ValueError(
            f"{name} is required; phosphate incorporation uses explicit "
            "DFT-derived spin-coupling tensors, not fixed CNOT/CZ gates"
        )
    missing = [key for key in _INCORPORATION_PAIR_KEYS if key not in tensors]
    if missing:
        raise ValueError(f"{name} is missing pair tensors: {', '.join(missing)}")
    return {
        key: _require_spin_tensor(f"{name}.{key}", dict(tensors[key]))
        for key in _INCORPORATION_PAIR_KEYS
    }


def _validate_rate(name: str, value: float) -> float:
    rate = float(value)
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {rate}")
    return rate


def _apply_spin_half_tensor(
    qc,
    q_a: int,
    q_b: int,
    tensor: dict[str, float],
    angle: float,
) -> None:
    """Apply symmetric spin-1/2 coupling tensor as Pauli product rotations."""
    qc.rxx(tensor["Axx"] * angle, q_a, q_b)
    qc.ryy(tensor["Ayy"] * angle, q_a, q_b)
    qc.rzz(tensor["Azz"] * angle, q_a, q_b)
    axy = tensor["Axy"]
    axz = tensor["Axz"]
    ayz = tensor["Ayz"]
    if axy:
        qc.sdg(q_b)
        qc.rxx(axy * angle, q_a, q_b)
        qc.s(q_b)
        qc.sdg(q_a)
        qc.rxx(axy * angle, q_a, q_b)
        qc.s(q_a)
    if axz:
        qc.h(q_b)
        qc.rxx(axz * angle, q_a, q_b)
        qc.h(q_b)
        qc.h(q_a)
        qc.rxx(axz * angle, q_a, q_b)
        qc.h(q_a)
    if ayz:
        qc.sdg(q_a)
        qc.h(q_b)
        qc.rxx(ayz * angle, q_a, q_b)
        qc.s(q_a)
        qc.h(q_b)
        qc.h(q_a)
        qc.sdg(q_b)
        qc.rxx(ayz * angle, q_a, q_b)
        qc.h(q_a)
        qc.s(q_b)


def _require_ca43_inputs(
    ca43_hf_tensors: dict[int, dict[str, float]] | None,
    ca_electron_map: dict[int, int] | None,
) -> tuple[dict[int, dict[str, float]], dict[int, int]]:
    if ca43_hf_tensors is None:
        raise ValueError("ca43_hf_tensors is required; no ⁴³Ca hyperfine estimate is bundled")
    if ca_electron_map is None:
        raise ValueError("ca_electron_map is required; no proximity-based electron map is bundled")
    tensors = {
        int(k): _require_spin_tensor(f"ca43_hf_tensors[{k}]", dict(v))
        for k, v in ca43_hf_tensors.items()
    }
    mapping = {int(k): int(v) for k, v in ca_electron_map.items()}
    expected = set(_CA43_START_QUBITS)
    if set(tensors) != expected:
        raise ValueError(f"ca43_hf_tensors keys must be {sorted(expected)}")
    if set(mapping) != expected:
        raise ValueError(f"ca_electron_map keys must be {sorted(expected)}")
    invalid = {k: v for k, v in mapping.items() if v not in (0, 1)}
    if invalid:
        raise ValueError(f"ca_electron_map values must be electron qubits 0 or 1, got {invalid}")
    return tensors, mapping


def _extended_arg(args: Any, key: str, default: Any = None) -> Any:
    value = getattr(args, key, None)
    if value is not None:
        return value
    params = getattr(args, "extended_params", None) or {}
    return params.get(key, default)


def _transport_rate_for_delay(args: Any, delay: int) -> float | None:
    if delay == 0:
        return None
    rates = _extended_arg(args, "transport_depolarizing_rates")
    if rates is None:
        raise ValueError("transport_depolarizing_rates is required for nonzero transport delays")
    if delay in rates:
        return float(rates[delay])
    delay_key = str(delay)
    if delay_key in rates:
        return float(rates[delay_key])
    raise ValueError(f"transport_depolarizing_rates lacks entry for delay {delay}dt")


def _spin_half_ops() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    return sx, sy, sz


def _spin_i_ops(I_spin: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dim = int(2 * I_spin + 1)
    m_vals = np.arange(-I_spin, I_spin + 1, dtype=float)
    iz = np.diag(m_vals).astype(complex)
    ip = np.zeros((dim, dim), dtype=complex)
    for idx, m in enumerate(m_vals[:-1]):
        ip[idx + 1, idx] = math.sqrt(I_spin * (I_spin + 1) - m * (m + 1))
    im = ip.conj().T
    ix = 0.5 * (ip + im)
    iy = -0.5j * (ip - im)
    return ix, iy, iz


def _ca43_unitary(tensor: dict[str, float], angle: float) -> np.ndarray:
    from scipy.linalg import expm as _expm

    sx, sy, sz = _spin_half_ops()
    ix, iy, iz = _spin_i_ops(3.5)
    H = (
        tensor["Axx"] * np.kron(sx, ix)
        + tensor["Ayy"] * np.kron(sy, iy)
        + tensor["Azz"] * np.kron(sz, iz)
        + tensor["Axy"] * (np.kron(sx, iy) + np.kron(sy, ix))
        + tensor["Axz"] * (np.kron(sx, iz) + np.kron(sz, ix))
        + tensor["Ayz"] * (np.kron(sy, iz) + np.kron(sz, iy))
    )
    return _expm(-1j * H * angle)


# ═════════════════════════════════════════════════════════════════
# Biological Noise Model (310K)
# ═════════════════════════════════════════════════════════════════


def biological_noise_model(
    T_celsius: float = 37.0,
    n_qubits: int = 8,
    *,
    cage_dephasing_rate: float | None = None,
):
    """Construct a Qiskit noise model calibrated to biological Posner.

    Supports 8q, 16q, and 35q circuit layouts with correct
    electron/nuclear qubit assignments for each.

    Noise channels based on measured ³¹P NMR relaxation in aqueous
    phosphate solutions at physiological temperature.

    Electron spins: radical lifetime T₁ ~ 1 µs, T₂ ~ 0.5 µs.
    Nuclear spins (³¹P): T₁ ~ 5 s, T₂ ~ 0.5 s in PO₄ solution.
    Ref: Maudsley, J. Magn. Reson. 69 (1986) 488.
         Hahn, Phys. Rev. 80 (1950) 580 (spin echo T₂).

    Thermal population: p₁ = 1/(1 + exp(ℏω/kT)).
    At 310K and Earth's field (50 µT), ℏω/kT ~ 10⁻¹⁰ for ³¹P,
    so p₁ ≈ 0.5 (essentially maximally mixed).

    The gate_time parameter converts physical relaxation rates into
    per-gate error rates. We assume 1 Trotter step corresponds to
    ~1 ns of physical radical-pair evolution (from the HF coupling
    timescale: 1/a_iso ≈ 1/3540 MHz ≈ 0.28 ns).

    Args:
        n_qubits: Circuit layout (8, 16, or 35).
            8q:  [0,1]=electrons, [2..7]=³¹P nuclei
            16q: [0,1,8,9]=electrons, [2..7,10..15]=³¹P nuclei
            35q: [0,1]=electrons, [2..7]=³¹P, [8..34]=⁴³Ca (3q each)
    """
    from qiskit_aer.noise import NoiseModel, thermal_relaxation_error

    # Physical relaxation parameters
    T1_electron = 1e-6  # 1 µs  — radical electron in solution
    T2_electron = 0.5e-6  # 0.5 µs — electron spin dephasing
    T1_nuclear = 5.0  # 5 s   — ³¹P longitudinal relaxation
    T2_nuclear = 0.5  # 0.5 s — ³¹P transverse relaxation

    # Gate time: one RXX/RYY/RZZ gate ≈ 1 ns of physical time
    gate_time = 1e-9  # seconds

    # Thermal population at 310K
    p_excited = 0.4999  # essentially 0.5 at biological field/temp

    noise = NoiseModel()

    # Electron error channel (fast: T₁=1µs, T₂=0.5µs)
    e_err_1q = thermal_relaxation_error(
        t1=T1_electron,
        t2=T2_electron,
        time=gate_time,
        excited_state_population=p_excited,
    )
    e_err_2q = e_err_1q.tensor(e_err_1q)

    # Nuclear error channel (slow: T₁=5s, T₂=0.5s)
    n_err_1q = thermal_relaxation_error(
        t1=T1_nuclear,
        t2=T2_nuclear,
        time=gate_time,
        excited_state_population=p_excited,
    )
    n_err_2q = n_err_1q.tensor(n_err_1q)

    # Mixed channel: electron-nuclear 2q gate
    en_err_2q = e_err_1q.tensor(n_err_1q)
    ne_err_2q = n_err_1q.tensor(e_err_1q)

    # Qubit assignments per circuit layout
    if n_qubits == 8:
        electron_qubits = [0, 1]
        nuclear_qubits = list(range(2, 8))
    elif n_qubits == 16:
        electron_qubits = [0, 1, 8, 9]
        nuclear_qubits = list(range(2, 8)) + list(range(10, 16))
    elif n_qubits == 35:
        electron_qubits = [0, 1]
        nuclear_qubits = list(range(2, 35))  # ³¹P + ⁴³Ca all nuclear
    else:
        raise ValueError(f"Unsupported n_qubits={n_qubits}. Use 8, 16, or 35.")

    for gate in ["rxx", "ryy", "rzz", "cx", "ecr"]:
        # Electron-electron gates
        for i, eq1 in enumerate(electron_qubits):
            for eq2 in electron_qubits[i + 1 :]:
                noise.add_quantum_error(e_err_2q, gate, [eq1, eq2])
        # Electron-nuclear gates (HF coupling)
        for eq in electron_qubits:
            for nq in nuclear_qubits:
                noise.add_quantum_error(en_err_2q, gate, [eq, nq])
                noise.add_quantum_error(ne_err_2q, gate, [nq, eq])
        # Nuclear-nuclear gates (dipolar)
        for i, nq1 in enumerate(nuclear_qubits):
            for nq2 in nuclear_qubits[i + 1 :]:
                noise.add_quantum_error(n_err_2q, gate, [nq1, nq2])

    # Posner cage vibrational dephasing: additional T₂ channel. This is an
    # external input, not inferred in code.
    if cage_dephasing_rate is None:
        raise ValueError(
            "cage_dephasing_rate is required; no measured Posner cage "
            "vibrational dephasing value is bundled"
        )
    from qiskit_aer.noise import phase_damping_error

    cage_dephasing = phase_damping_error(float(cage_dephasing_rate))
    for nq in nuclear_qubits:
        for gate_1q in ["rz", "x", "h"]:
            noise.add_quantum_error(cage_dephasing, gate_1q, [nq])

    return noise


def get_noise_params_dict(T_celsius: float = 37.0) -> dict[str, Any]:
    """Return biological noise parameters as a dictionary (no qiskit needed)."""
    T_K = T_celsius + 273.15
    return {
        "temperature_K": T_K,
        "T1_electron_s": 1e-6,
        "T2_electron_s": 0.5e-6,
        "T1_nuclear_s": 5.0,
        "T2_nuclear_s": 0.5,
        "gate_time_s": 1e-9,
        "p_excited": 0.4999,
        "cage_dephasing_rate": None,
        "note": "Cage dephasing is not bundled; pass an explicit measured value.",
    }


# ═════════════════════════════════════════════════════════════════
# Two-Posner Transport Circuit (16 qubits)
# ═════════════════════════════════════════════════════════════════


def build_two_posner_transport_circuit(
    J: float = 1.0,
    omega_0: float = 0.5,
    t_evolve: float = math.pi,
    n_trotter: int = 5,
    transport_delay_dt: int = 10000,
    dd_during_transport: bool = True,
    hf_site1: list[dict] | None = None,
    hf_site2: list[dict] | None = None,
    hf_site1_b: list[dict] | None = None,
    hf_site2_b: list[dict] | None = None,
    incorporation_tensors: dict[str, dict[str, float]] | None = None,
    incorporation_tensors_b: dict[str, dict[str, float]] | None = None,
    transport_depolarizing_rate: float | None = None,
):
    r"""16-qubit two-Posner entanglement swapping circuit.

    Full 8q per Posner: 2 electrons + 6 nuclei (3 per site).

    Qubit map (each Posner = 8 qubits):
      Posner A: q0=e₁, q1=e₂, q2=P₁*, q3=P₂, q4=P₃, q5=P₄, q6=P₅, q7=P₆
      Posner B: q8=e₃, q9=e₄, q10=P₇*, q11=P₈, q12=P₉, q13=P₁₀, q14=P₁₁, q15=P₁₂
      (* = entangled pair from PPᵢ hydrolysis)

    Args:
        dd_during_transport: If True, apply XY4 dynamical decoupling
            during the transport delay (reduces idle decoherence on QPU).
        hf_site1: List of 3 HF dicts {Axx,Ayy,Azz,Axy,Axz,Ayz} for
            Posner A site 1 nuclei.
        hf_site2: List of 3 HF dicts for Posner A site 2 nuclei.
        hf_site1_b: List of 3 HF dicts for Posner B site 1 nuclei.
            If None, defaults to hf_site1 (identical Posner).
        hf_site2_b: List of 3 HF dicts for Posner B site 2 nuclei.
            If None, defaults to hf_site2 (identical Posner).
        incorporation_tensors: Explicit spin-coupling tensors for the
            p1_p2, p1_p3, and p2_p3 incorporation pairs in Posner A.
        incorporation_tensors_b: Explicit Posner B incorporation tensors.
            If None, uses incorporation_tensors for identical Posners.
        transport_depolarizing_rate: Explicit calibrated nuclear
            depolarising probability for the requested transport delay.
    """
    from qiskit import QuantumCircuit

    hf1_a = _require_three_hf("hf_site1", hf_site1)
    hf2_a = _require_three_hf("hf_site2", hf_site2)
    # Posner B defaults to same as A (identical Ca₉(PO₄)₆ crystal)
    # unless explicitly overridden for asymmetric experiments.
    hf1_b = _require_three_hf("hf_site1_b", hf_site1_b) if hf_site1_b is not None else hf1_a
    hf2_b = _require_three_hf("hf_site2_b", hf_site2_b) if hf_site2_b is not None else hf2_a
    inc_a = _require_incorporation_tensors("incorporation_tensors", incorporation_tensors)
    inc_b = (
        _require_incorporation_tensors("incorporation_tensors_b", incorporation_tensors_b)
        if incorporation_tensors_b is not None
        else inc_a
    )

    qc = QuantumCircuit(16, 16)

    # ── Phase 1: PPi hydrolysis → entangled singlet (P₁-P₇) ──
    qc.x(10)
    qc.h(2)
    qc.cx(2, 10)
    qc.z(2)  # singlet on (q2, q10)

    # ── Phase 2: Electron singlet prep + incorporation ──
    # Posner A electrons
    qc.x(1)
    qc.h(0)
    qc.cx(0, 1)
    qc.z(0)
    # Posner B electrons
    qc.x(9)
    qc.h(8)
    qc.cx(8, 9)
    qc.z(8)

    # Incorporation: explicit phosphate-condensation spin Hamiltonian.
    # The previous fixed CNOT/partial-SWAP/CZ surrogate has been removed;
    # every pair angle below comes from caller-supplied coupling tensors.
    _apply_spin_half_tensor(qc, 2, 3, inc_a["p1_p2"], 1.0)
    _apply_spin_half_tensor(qc, 2, 4, inc_a["p1_p3"], 1.0)
    _apply_spin_half_tensor(qc, 3, 4, inc_a["p2_p3"], 1.0)
    _apply_spin_half_tensor(qc, 10, 11, inc_b["p1_p2"], 1.0)
    _apply_spin_half_tensor(qc, 10, 12, inc_b["p1_p3"], 1.0)
    _apply_spin_half_tensor(qc, 11, 12, inc_b["p2_p3"], 1.0)

    qc.barrier()

    # ── Phase 3: Full 8q Hamiltonian evolution ──
    # TRUE 2nd-order Suzuki-Trotter with full anisotropic HF tensors.
    # Error per step: O(dt³) vs O(dt²) for 1st-order.
    dt = t_evolve / n_trotter

    def _apply_exchange(qc, e0, e1, angle):
        qc.rxx(angle, e0, e1)
        qc.ryy(angle, e0, e1)
        qc.rzz(angle, e0, e1)

    def _apply_hf_full(qc, e_qubit, nuc_qubits, hf_list, angle_factor):
        """Apply full anisotropic HF: diagonal + off-diagonal cross terms."""
        for k, hf in enumerate(hf_list):
            nq = nuc_qubits[k]
            _apply_spin_half_tensor(qc, e_qubit, nq, hf, angle_factor)

    # Pre-compute dipolar table ONCE (not per Trotter step)
    try:
        from orca_posner_hf import compute_qubit_dipolar_tensor_table

        _16q_dip_table = compute_qubit_dipolar_tensor_table()
    except ImportError:
        raise RuntimeError(
            "orca_posner_hf.compute_qubit_dipolar_tensor_table is required for 16q "
            "nuclear dipolar tensor couplings"
        ) from None

    for step in range(n_trotter):
        for posner_offset in [0, 8]:
            e0, e1 = posner_offset, posner_offset + 1
            s1_nucs = [posner_offset + 2, posner_offset + 3, posner_offset + 4]
            s2_nucs = [posner_offset + 5, posner_offset + 6, posner_offset + 7]

            # Select HF tensors for this Posner:
            # Posner A (offset=0) uses hf1_a/hf2_a,
            # Posner B (offset=8) uses hf1_b/hf2_b.
            # Default: identical (same Ca₉(PO₄)₆ crystal structure).
            # Override hf_site1_b/hf_site2_b for asymmetric experiments
            # (e.g. isotope-enriched Posner B, defective crystal, etc.).
            if posner_offset == 0:
                _hf1, _hf2 = hf1_a, hf2_a
            else:
                _hf1, _hf2 = hf1_b, hf2_b

            # ── Forward sandwich: H_exch(dt/4) · H_hf(dt/4) ──
            _apply_exchange(qc, e0, e1, J * dt / 4)
            _apply_hf_full(qc, e0, s1_nucs, _hf1, dt / 4)
            _apply_hf_full(qc, e1, s2_nucs, _hf2, dt / 4)

            # ── Center: H_Z(dt) ──
            qc.rz(omega_0 * dt, e0)
            qc.rz(omega_0 * dt, e1)

            # ── Reverse sandwich: H_hf(dt/4) · H_exch(dt/4) ──
            _apply_hf_full(qc, e0, s1_nucs, _hf1, dt / 4)
            _apply_hf_full(qc, e1, s2_nucs, _hf2, dt / 4)
            _apply_exchange(qc, e0, e1, J * dt / 4)

        # ── Nuclear dipolar (per-pair from S₆ geometry) ──
        for posner_offset in [0, 8]:
            for row in _16q_dip_table:
                pi = int(row["qubit_i"]) + posner_offset
                pj = int(row["qubit_j"]) + posner_offset
                tensor = {key: float(row[key]) for key in _SPIN_TENSOR_KEYS}
                _apply_spin_half_tensor(qc, pi, pj, tensor, dt / 2)

    qc.barrier()

    # ── Phase 4: Transport with diffusion-induced decoherence ──
    # Physical model: Posner diffuses through cytoplasm.
    # Brownian motion: D ≈ 10⁻¹⁰ m²/s for ~1nm cluster in water.
    # Transport time: t = L²/(2D). For L=10µm: t ≈ 0.5s.
    #
    # During transport, environmental coupling causes decoherence.
    # We model this as:
    #   (a) QPU: delay + DD (hardware error mitigation)
    #   (b) Physics: partial depolarizing channel after delay
    #       (amplitude ∝ transport_delay, modeling collisional dephasing)
    if transport_delay_dt > 0:
        if dd_during_transport:
            # XY4 dynamical decoupling during transport (QPU-specific)
            # NOTE: DD is a QPU error mitigation technique, NOT a
            # biological process. In the cell there is no DD.
            # π-pulses on ELECTRON qubits ONLY — applying X/Y to nuclear
            # qubits would flip the nuclear spin state and corrupt the
            # physics. Ref: verify_ibm_heron.py lines 399-412.
            electron_qubits = [0, 1, 8, 9]
            dd_interval = transport_delay_dt // 5
            # All qubits idle during DD intervals
            for q in range(16):
                qc.delay(dd_interval, q, unit="dt")
            for q in electron_qubits:
                qc.x(q)
            for q in range(16):
                qc.delay(dd_interval, q, unit="dt")
            for q in electron_qubits:
                qc.y(q)
            for q in range(16):
                qc.delay(dd_interval, q, unit="dt")
            for q in electron_qubits:
                qc.x(q)
            for q in range(16):
                qc.delay(dd_interval, q, unit="dt")
            for q in electron_qubits:
                qc.y(q)
            for q in range(16):
                qc.delay(dd_interval, q, unit="dt")
        else:
            for q in range(16):
                qc.delay(transport_delay_dt, q, unit="dt")

        # Diffusion-induced decoherence: proper depolarising noise channel.
        # This MUST be supplied as a calibrated probability. The code no
        # longer derives a rate from an unvalidated Brownian surrogate.
        if transport_depolarizing_rate is None:
            raise ValueError("transport_depolarizing_rate is required when transport_delay_dt > 0")
        from qiskit_aer.noise import NoiseModel, depolarizing_error

        _transport_noise = NoiseModel()
        p_depol = _validate_rate("transport_depolarizing_rate", transport_depolarizing_rate)
        depol_1q = depolarizing_error(p_depol, 1)
        for q in list(range(2, 8)) + list(range(10, 16)):  # nuclear qubits only
            _transport_noise.add_quantum_error(depol_1q, "rz", [q])
            _transport_noise.add_quantum_error(depol_1q, "id", [q])
        # Store on circuit metadata so callers can access it
        qc.metadata = qc.metadata or {}
        qc.metadata["transport_noise_model"] = _transport_noise

    qc.barrier()

    # ── Phase 5: Electron singlet unmeasurement (both Posners) ──
    qc.cx(0, 1)
    qc.h(0)
    qc.cx(8, 9)
    qc.h(8)

    # ── Phase 6: Binding — Bell measurement on entangled pair ──
    qc.cx(2, 10)
    qc.h(2)

    # ── Phase 7: Measure all ──
    qc.measure(range(16), range(16))

    return qc


def run_two_posner_simulation(shots: int = 4096, **circuit_kwargs) -> tuple:
    """Build and run 16q transport circuit WITH noise model applied.

    Returns (counts, analysis_dict). Uses AerSimulator with the
    transport depolarizing noise model automatically wired in.
    """
    from qiskit import transpile

    qc = build_two_posner_transport_circuit(**circuit_kwargs)
    noise_model = None
    if qc.metadata and "transport_noise_model" in qc.metadata:
        noise_model = qc.metadata["transport_noise_model"]
    try:
        from qiskit_aer import AerSimulator

        sim = AerSimulator(noise_model=noise_model)
        tqc = transpile(qc, sim)
        counts = sim.run(tqc, shots=shots).result().get_counts()
    except ImportError:
        if noise_model is not None:
            raise RuntimeError(
                "qiskit-aer is required because transport noise is present; "
                "refusing noiseless statevector fallback"
            )
        from qiskit.quantum_info import Statevector

        sv = Statevector.from_instruction(qc.remove_final_measurements(inplace=False))
        counts = sv.sample_counts(shots)
    analysis = analyse_two_posner(counts, shots)
    return counts, analysis


def analyse_two_posner(counts: dict, shots: int | None = None) -> dict:
    """Analyse two-Posner transport circuit results.

    Returns:
    - binding_probability: P(singlet on q2,q10) = P(both = 1 in Bell basis)
    - posner_a_singlet: P(electron singlet in Posner A)
    - posner_b_singlet: P(electron singlet in Posner B)
    - nuclear_correlation: ⟨Z₃Z₁₁⟩ (non-entangled nuclei, control)
    """
    total = shots or sum(counts.values())

    def bit(bs, q):
        return int(bs.replace(" ", "")[-(q + 1)])

    # Binding: P(q2=1 AND q10=1 in Bell basis) → singlet survived
    n_bind = sum(c for bs, c in counts.items() if bit(bs, 2) == 1 and bit(bs, 10) == 1)
    p_bind = n_bind / total if total else 0

    # Posner A electron singlet: q0=1, q1=1 (after CX+H unmeasurement)
    n_ea = sum(c for bs, c in counts.items() if bit(bs, 0) == 1 and bit(bs, 1) == 1)
    p_ea = n_ea / total if total else 0

    # Posner B electron singlet: q8=1, q9=1
    n_eb = sum(c for bs, c in counts.items() if bit(bs, 8) == 1 and bit(bs, 9) == 1)
    p_eb = n_eb / total if total else 0

    # Nuclear correlation ⟨Z₃Z₁₁⟩ (P₂ in A vs P₅ in B — should be uncorrelated)
    n_same = sum(c for bs, c in counts.items() if bit(bs, 3) == bit(bs, 11))
    zz_nucl = (2 * n_same - total) / total if total else 0

    return {
        "binding_probability": round(p_bind, 6),
        "posner_a_electron_singlet": round(p_ea, 6),
        "posner_b_electron_singlet": round(p_eb, 6),
        "nuclear_correlation_zz": round(zz_nucl, 6),
        "shots": total,
    }


# ═════════════════════════════════════════════════════════════════
# ⁴³Ca-Enriched Posner (35 qubits)
# ═════════════════════════════════════════════════════════════════
#
# I=7/2 (⁴³Ca) → 3 qubits per calcium using binary encoding:
#   |m=-7/2⟩ = |000⟩, |m=-5/2⟩ = |001⟩, ..., |m=+7/2⟩ = |111⟩
#
# Qubit map (35 total):
#   q0, q1: electrons
#   q2-q4: ³¹P site 1 (I=1/2, 1 qubit each)
#   q5-q7: ³¹P site 2
#   q8-q10: ⁴³Ca₁ (I=7/2, 3 qubits)
#   q11-q13: ⁴³Ca₂
#   q14-q16: ⁴³Ca₃
#   q17-q19: ⁴³Ca₄
#   q20-q22: ⁴³Ca₅
#   q23-q25: ⁴³Ca₆
#   q26-q28: ⁴³Ca₇
#   q29-q31: ⁴³Ca₈
#   q32-q34: ⁴³Ca₉ (central Ca)


def build_posner_43ca_circuit(
    J: float = 1.0,
    omega_0: float = 0.5,
    t: float = math.pi,
    n_trotter: int = 3,
    p31_hf_site1: list[dict] | None = None,
    p31_hf_site2: list[dict] | None = None,
    ca43_hf_tensors: dict[int, dict[str, float]] | None = None,
    ca_electron_map: dict[int, int] | None = None,
) -> Any:
    """35-qubit ⁴³Ca-enriched Posner circuit.

    Implements full electron--I=7/2 Ca hyperfine coupling via exact
    16×16 UnitaryGate matrix exponentiation for each Ca tensor.

    ³¹P nuclei use full anisotropic 6-component HF tensors
    matching the 8q/16q circuits for consistency.
    """
    from qiskit import QuantumCircuit

    hf1 = _require_three_hf("p31_hf_site1", p31_hf_site1)
    hf2 = _require_three_hf("p31_hf_site2", p31_hf_site2)
    ca_tensors, ca_map = _require_ca43_inputs(ca43_hf_tensors, ca_electron_map)

    qc = QuantumCircuit(35, 35)

    # Electron singlet prep
    qc.x(1)
    qc.h(0)
    qc.cx(0, 1)
    qc.z(0)

    dt = t / n_trotter

    # Helper: apply full anisotropic ³¹P HF with off-diagonal cross-coupling
    def _apply_p31_hf(qc, e_qubit, nuc_qubits, hf_list, angle_factor):
        for k, hf in enumerate(hf_list):
            nq = nuc_qubits[k]
            _apply_spin_half_tensor(qc, e_qubit, nq, hf, angle_factor)

    from qiskit.circuit.library import UnitaryGate

    _ca_gates = {
        ca_start: UnitaryGate(_ca43_unitary(tensor, dt / 4), label=f"Ca{ca_start}-e HF")
        for ca_start, tensor in ca_tensors.items()
    }

    # Pre-compute dipolar table ONCE
    try:
        from orca_posner_hf import compute_qubit_dipolar_tensor_table

        _35q_dip_table = compute_qubit_dipolar_tensor_table()
    except ImportError:
        raise RuntimeError(
            "orca_posner_hf.compute_qubit_dipolar_tensor_table is required for 35q "
            "nuclear dipolar tensor couplings"
        ) from None

    for _ in range(n_trotter):
        # ── 2nd-order Suzuki-Trotter: forward sandwich ──
        qc.rxx(J * dt / 4, 0, 1)
        qc.ryy(J * dt / 4, 0, 1)
        qc.rzz(J * dt / 4, 0, 1)

        # H_P31_hf(dt/4) — full anisotropic with cross terms
        _apply_p31_hf(qc, 0, [2, 3, 4], hf1, dt / 4)
        _apply_p31_hf(qc, 1, [5, 6, 7], hf2, dt / 4)

        # Full ⁴³Ca tensor hyperfine, exact 16×16 unitary.
        for ca_start, e_qubit in ca_map.items():
            q0c, q1c, q2c = ca_start, ca_start + 1, ca_start + 2
            qc.append(_ca_gates[ca_start], [e_qubit, q2c, q1c, q0c])

        # ── Center: Electron Zeeman (dt) ──
        qc.rz(omega_0 * dt, 0)
        qc.rz(omega_0 * dt, 1)

        # ── 2nd-order: reverse sandwich ──
        # Full ⁴³Ca tensor hyperfine reverse half.
        for ca_start, e_qubit in ca_map.items():
            q0c, q1c, q2c = ca_start, ca_start + 1, ca_start + 2
            qc.append(_ca_gates[ca_start], [e_qubit, q2c, q1c, q0c])

        # H_P31_hf(dt/4) — reverse order
        _apply_p31_hf(qc, 1, [5, 6, 7], hf2, dt / 4)
        _apply_p31_hf(qc, 0, [2, 3, 4], hf1, dt / 4)

        # H_exch(dt/4)
        qc.rxx(J * dt / 4, 0, 1)
        qc.ryy(J * dt / 4, 0, 1)
        qc.rzz(J * dt / 4, 0, 1)

        # ── Nuclear dipolar (³¹P pairs, pre-computed table) ──
        for row in _35q_dip_table:
            qi = int(row["qubit_i"])
            qj = int(row["qubit_j"])
            tensor = {key: float(row[key]) for key in _SPIN_TENSOR_KEYS}
            _apply_spin_half_tensor(qc, qi, qj, tensor, dt / 2)

    # Unmeasure electron singlet
    qc.cx(0, 1)
    qc.h(0)
    qc.measure(range(35), range(35))
    return qc


def analyse_43ca(counts: dict) -> dict:
    """Analyse 35q ⁴³Ca Posner results."""
    total = sum(counts.values())

    def bit(bs, q):
        return int(bs.replace(" ", "")[-(q + 1)])

    # Electron singlet
    ns = sum(c for bs, c in counts.items() if bit(bs, 0) == 1 and bit(bs, 1) == 1)
    p_singlet = ns / total if total else 0

    # Ca spin polarization: sum of Iz for each Ca (binary decode)
    ca_pol = []
    for ca_start in range(8, 35, 3):
        iz_sum = 0.0
        for bs, c in counts.items():
            m = 4 * bit(bs, ca_start) + 2 * bit(bs, ca_start + 1) + bit(bs, ca_start + 2)
            iz_sum += (m - 3.5) * c  # center: m ∈ [0,7] → Iz ∈ [-3.5, 3.5]
        ca_pol.append(round(iz_sum / total, 4) if total else 0)

    return {
        "singlet_probability": round(p_singlet, 6),
        "ca_polarizations": ca_pol,
        "shots": total,
    }


# ═════════════════════════════════════════════════════════════════
# Experiment Runners (for integration with verify_ibm_heron.py)
# ═════════════════════════════════════════════════════════════════


def run_exp4_biological_noise(build_posner_circuit, _exec_fn, args, P=print):
    """Exp 4: Compare ideal vs biological noise predictions.

    Runs the same Posner circuit under:
    1. Ideal (statevector) — no noise
    2. Biological noise model (310K) — simulated decoherence
    """
    from qiskit.quantum_info import Statevector
    from qiskit_aer import AerSimulator

    P("\n▸ Exp 4: Biological noise comparison (310K vs ideal)")
    results = {}

    qc = build_posner_circuit(
        J=1.0,
        omega_0=0.5,
        t=math.pi,
        n_trotter=5,
        hf1=args.hf1,
        hf2=args.hf2,
    )

    # 4a: Ideal (statevector)
    sv = Statevector.from_instruction(qc.remove_final_measurements(inplace=False))
    counts_ideal = sv.sample_counts(args.shots)
    from verify_ibm_heron import analyse_rpm_8q

    r_ideal = analyse_rpm_8q(counts_ideal)
    P(f"  Ideal (no noise):     Φ_S = {r_ideal['singlet_probability']:.4f}")
    results["ideal"] = r_ideal

    # 4b: Biological noise (310K)
    try:
        bio_noise = biological_noise_model(
            37.0,
            cage_dephasing_rate=_extended_arg(args, "cage_dephasing_rate"),
        )
        sim_bio = AerSimulator(noise_model=bio_noise)
        from qiskit import transpile

        tqc = transpile(qc, sim_bio)
        counts_bio = sim_bio.run(tqc, shots=args.shots).result().get_counts()
        r_bio = analyse_rpm_8q(counts_bio)
        P(f"  Biological (310K):    Φ_S = {r_bio['singlet_probability']:.4f}")
        results["biological_310K"] = r_bio
    except ImportError:
        P("  [SKIP] qiskit-aer not available for noise simulation")
        results["biological_310K"] = {"error": "qiskit-aer not available"}

    # 4c: Summary
    if "singlet_probability" in results.get("biological_310K", {}):
        delta = abs(r_ideal["singlet_probability"] - r_bio["singlet_probability"])
        P(f"  Δ(ideal - bio):      {delta:.4f}")
        P("  Note: small Δ expected (nuclear T₁=5s ≫ gate time ≈1ns)")
        results["delta_ideal_bio"] = round(delta, 6)

    return results


def run_exp5_transport(args, P=print):
    """Exp 5: Two-Posner entanglement transport.

    Uses run_two_posner_simulation() which automatically applies
    the transport depolarizing noise model via AerSimulator.
    """
    P("\n▸ Exp 5: Two-Posner entanglement transport (16q)")
    results = {}

    transport_delays = [0, 1000, 5000, 10000, 50000]
    for delay in transport_delays:
        counts, r = run_two_posner_simulation(
            shots=args.shots,
            J=1.0,
            omega_0=0.5,
            transport_delay_dt=delay,
            hf_site1=args.hf1,
            hf_site2=args.hf2,
            incorporation_tensors=_extended_arg(args, "incorporation_tensors"),
            incorporation_tensors_b=_extended_arg(args, "incorporation_tensors_b"),
            transport_depolarizing_rate=_transport_rate_for_delay(args, delay),
        )
        P(
            f"  transport={delay:>5d}dt  P(bind)={r['binding_probability']:.4f}"
            f"  ⟨Z₃Z₁₁⟩={r['nuclear_correlation_zz']:.4f}"
        )
        results[f"delay_{delay}dt"] = r

    return results


def run_exp6_43ca(args, P=print):
    """Exp 6: ⁴³Ca-enriched Posner (35q, MPS simulator)."""
    P("\n▸ Exp 6: ⁴³Ca-enriched Posner (35q, MPS simulator)")
    results = {}

    try:
        from qiskit_aer import AerSimulator
        from qiskit import transpile

        sim = AerSimulator(method="matrix_product_state")
    except ImportError:
        P("  [SKIP] qiskit-aer not available for MPS simulation")
        return {"error": "qiskit-aer not available"}

    for J in [0.5, 1.0, 5.0]:
        qc = build_posner_43ca_circuit(
            J=J,
            omega_0=0.5,
            n_trotter=3,
            p31_hf_site1=args.hf1,
            p31_hf_site2=args.hf2,
            ca43_hf_tensors=_extended_arg(args, "ca43_hf_tensors"),
            ca_electron_map=_extended_arg(args, "ca_electron_map"),
        )
        tqc = transpile(qc, sim)
        try:
            res = sim.run(tqc, shots=args.shots).result()
            counts = res.get_counts()
            r = analyse_43ca(counts)
            P(
                f"  J={J:.1f}  Φ_S={r['singlet_probability']:.4f}"
                f"  ⟨Iz_Ca1⟩={r['ca_polarizations'][0]:.3f}"
            )
            results[f"J_{J}"] = r
        except Exception as e:
            P(f"  J={J:.1f}  ERROR: {e}")
            results[f"J_{J}"] = {"error": str(e)}

    return results

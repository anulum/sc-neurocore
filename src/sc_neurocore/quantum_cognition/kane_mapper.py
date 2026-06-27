# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Kane architecture silicon mapper

r"""Kane-architecture Si:P qubit register layout mapper.

Maps the abstract :class:`SpinPoolMPS` (non-local spin sites) to a
physical layout of phosphorus-31 donor atoms embedded in a silicon
lattice, following the Kane (1998) quantum computing architecture.

This is a **design-space exploration tool** — no hardware dependencies.
It computes physical qubit placement, inter-donor exchange coupling
strengths, and decoherence budgets for a given spin pool configuration.

Architecture overview
---------------------
In the Kane architecture, individual :sup:`31`\P atoms are implanted
~20 nm below the Si surface, spaced ~20–50 nm apart.  Each donor's
nuclear spin serves as a data qubit.  Control is via:

- **A-gates** (metal electrodes above each donor): tune the hyperfine
  coupling between the electron and nuclear spin.
- **J-gates** (between adjacent donors): tune the exchange coupling
  between neighbouring electron spins, enabling two-qubit gates.

The exchange coupling *J(d)* between two donors separated by distance
*d* decays exponentially:

.. math::

    J(d) = J_0 \exp\!\left(-\frac{2d}{a_B^*}\right)

where :math:`a_B^* \approx 2.5\,\text{nm}` is the effective Bohr
radius of the donor electron in silicon and :math:`J_0 \approx 0.1`
meV.

Decoherence
-----------
- Nuclear spin T₂ in enriched ²⁸Si: >30 s (Muhonen et al., 2014)
- Electron spin T₂ at 1 K: ~2 ms (standard), ~1 s (dynamical decoupling)
- Gate time: ~10 ns (electron), ~1 µs (nuclear)

References
----------
- Kane, B. E. "A silicon-based nuclear spin quantum computer."
  *Nature* 393, 133–137 (1998). doi:10.1038/30156
- Muhonen, J. T. et al. "Storing quantum information for 30 seconds
  in a nanoelectronic device." *Nature Nanotech.* 9 (2014).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Physical constants for Si:P system
_BOHR_RADIUS_STAR_NM = 2.5  # effective Bohr radius in Si [nm]
_J0_MEV = 0.1  # exchange coupling prefactor [meV]
_DEFAULT_SPACING_NM = 20.0  # default inter-donor spacing [nm]
_T2_NUCLEAR_S = 30.0  # nuclear spin T₂ in ²⁸Si [s]
_T2_ELECTRON_MS = 2.0  # electron spin T₂ at 1 K [ms]
_GATE_1Q_NS = 10.0  # single-qubit A-gate time [ns]
_GATE_2Q_NS = 50.0  # two-qubit J-gate time [ns]
_SWAP_COST_GATES = 3  # SWAP = 3 CX gates
_DEPTH_NM = 20.0  # donor implantation depth below surface [nm]

_VALID_TOPOLOGIES = ("linear", "grid", "triangular", "hexagonal")


@dataclass(frozen=True)
class KaneRegisterLayout:
    """Physical layout of a Si:P qubit register.

    Attributes
    ----------
    n_qubits : int
        Number of ³¹P donor qubits.
    qubit_positions : np.ndarray[Any, Any]
        Shape ``(n_qubits, 2)`` — (x, y) coordinates in nanometres.
    coupling_matrix : np.ndarray[Any, Any]
        Shape ``(n_qubits, n_qubits)`` — exchange coupling J(d) in meV.
        Symmetric, diagonal is zero.
    depth_nm : float
        Implantation depth below Si surface.
    t2_budget_ms : float
        T₂ decoherence budget for the register in milliseconds.
    max_gate_depth : int
        Maximum circuit depth achievable within the T₂ budget.
    gate_schedule : list[dict]
        Ordered list of gate operations with timing.
    """

    n_qubits: int
    qubit_positions: np.ndarray[Any, Any]
    coupling_matrix: np.ndarray[Any, Any]
    depth_nm: float = _DEPTH_NM
    t2_budget_ms: float = 0.0
    max_gate_depth: int = 0
    gate_schedule: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-compatible dict."""
        return {
            "n_qubits": self.n_qubits,
            "qubit_positions_nm": self.qubit_positions.tolist(),
            "coupling_matrix_meV": self.coupling_matrix.tolist(),
            "depth_nm": self.depth_nm,
            "t2_budget_ms": self.t2_budget_ms,
            "max_gate_depth": self.max_gate_depth,
        }


class KaneSiliconMapper:
    """Map SpinPoolMPS sites to a Kane-architecture Si:P register.

    Parameters
    ----------
    spacing_nm : float
        Target inter-donor spacing in nanometres (default 20 nm).
    depth_nm : float
        Implantation depth below silicon surface (default 20 nm).
    topology : str
        Layout topology: ``"linear"``, ``"grid"``, ``"triangular"``, or
        ``"hexagonal"``.
    """

    def __init__(
        self,
        spacing_nm: float = _DEFAULT_SPACING_NM,
        depth_nm: float = _DEPTH_NM,
        topology: str = "linear",
    ) -> None:
        if spacing_nm <= 0:
            raise ValueError(f"spacing_nm must be > 0, got {spacing_nm}")
        if topology not in _VALID_TOPOLOGIES:
            raise ValueError(f"topology must be one of {_VALID_TOPOLOGIES}, got {topology!r}")

        self.spacing_nm = spacing_nm
        self.depth_nm = depth_nm
        self.topology = topology

    def map_pool_to_register(self, n_sites: int) -> KaneRegisterLayout:
        """Compute physical qubit placement and coupling matrix.

        Parameters
        ----------
        n_sites : int
            Number of spin pool sites to map.

        Returns
        -------
        KaneRegisterLayout
            Physical register layout with coupling strengths and
            decoherence budget.
        """
        if n_sites < 1:
            raise ValueError(f"n_sites must be >= 1, got {n_sites}")

        positions = self._compute_positions(n_sites)
        coupling = self._compute_coupling_matrix(positions)
        t2_ms = _T2_ELECTRON_MS
        max_depth = int(t2_ms * 1e6 / _GATE_2Q_NS)
        schedule = self._build_gate_schedule(n_sites, coupling)

        layout = KaneRegisterLayout(
            n_qubits=n_sites,
            qubit_positions=positions,
            coupling_matrix=coupling,
            depth_nm=self.depth_nm,
            t2_budget_ms=t2_ms,
            max_gate_depth=max_depth,
            gate_schedule=schedule,
        )

        logger.info(
            "Kane register: %d qubits, topology=%s, spacing=%.1f nm, "
            "max_coupling=%.4f meV, T₂=%.1f ms, max_depth=%d, gates=%d",
            n_sites,
            self.topology,
            self.spacing_nm,
            float(np.max(coupling[coupling > 0])) if np.any(coupling > 0) else 0.0,
            t2_ms,
            max_depth,
            len(schedule),
        )

        return layout

    def _compute_positions(self, n: int) -> np.ndarray[Any, Any]:
        """Compute qubit positions based on topology."""
        if self.topology == "linear":
            positions = np.zeros((n, 2), dtype=np.float64)
            positions[:, 0] = np.arange(n) * self.spacing_nm
            return positions

        if self.topology == "grid":
            cols = int(np.ceil(np.sqrt(n)))
            positions = np.zeros((n, 2), dtype=np.float64)
            for i in range(n):
                positions[i, 0] = (i % cols) * self.spacing_nm
                positions[i, 1] = (i // cols) * self.spacing_nm
            return positions

        if self.topology == "triangular":
            # Equilateral triangular lattice
            cols = int(np.ceil(np.sqrt(n)))
            positions = np.zeros((n, 2), dtype=np.float64)
            for i in range(n):
                row = i // cols
                col = i % cols
                positions[i, 0] = (col + 0.5 * (row % 2)) * self.spacing_nm
                positions[i, 1] = row * self.spacing_nm * np.sqrt(3) / 2
            return positions

        # Hexagonal (honeycomb) lattice
        cols = max(int(np.ceil(np.sqrt(n))), 2)
        positions = np.zeros((n, 2), dtype=np.float64)
        idx = 0
        for row in range(n):
            for col in range(cols):
                if idx >= n:
                    break
                # Honeycomb: alternate sublattice offsets
                sub = (row + col) % 2
                positions[idx, 0] = col * self.spacing_nm * 1.5
                positions[idx, 1] = (
                    row * self.spacing_nm * np.sqrt(3) / 2 + sub * self.spacing_nm * np.sqrt(3) / 4
                )
                idx += 1
            if idx >= n:
                break
        return positions

    def _compute_coupling_matrix(self, positions: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Compute exchange coupling J(d) between all donor pairs."""
        n = len(positions)
        coupling = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(positions[i] - positions[j]))
                j_val = self._exchange_coupling(d)
                coupling[i, j] = j_val
                coupling[j, i] = j_val

        return coupling

    @staticmethod
    def _exchange_coupling(distance_nm: float) -> float:
        """Compute exchange coupling J(d) in meV.

        Uses the exponential decay model from Kane (1998):
        J(d) = J₀ · exp(-2d / a_B*)
        """
        if distance_nm <= 0:
            return _J0_MEV
        return float(_J0_MEV * np.exp(-2.0 * distance_nm / _BOHR_RADIUS_STAR_NM))

    def _build_gate_schedule(self, n: int, coupling: np.ndarray[Any, Any]) -> list[dict[str, Any]]:
        """Build a gate schedule with DAG-based parallel scheduling.

        Produces an optimized gate schedule where:
        - A-gates (1Q) execute in parallel on all qubits
        - J-gates (2Q) are grouped into parallel layers: gates on
          non-overlapping qubit pairs execute simultaneously
        - Layers are ordered by coupling strength (strongest first)
        """
        schedule = []
        t_ns = 0.0

        # Layer 0: parallel A-gates on all qubits
        for q in range(n):
            schedule.append(
                {
                    "gate": "A",
                    "qubits": [q],
                    "time_ns": t_ns,
                    "duration_ns": _GATE_1Q_NS,
                    "layer": 0,
                    "description": f"A-gate: init qubit {q}",
                }
            )
        t_ns += _GATE_1Q_NS

        # Collect all J-gate pairs sorted by coupling strength
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if coupling[i, j] > 1e-10:
                    pairs.append((i, j, coupling[i, j]))
        pairs.sort(key=lambda x: -x[2])

        # Greedy graph-coloring for parallel layers:
        # Two J-gates can execute in parallel if they share no qubits.
        remaining = list(pairs)
        layer_idx = 1
        while remaining:
            layer: list[tuple[int, int, float]] = []
            used_qubits: set[int] = set()
            still_remaining = []

            for qi, qj, j_val in remaining:
                if qi not in used_qubits and qj not in used_qubits:
                    layer.append((qi, qj, j_val))
                    used_qubits.add(qi)
                    used_qubits.add(qj)
                else:
                    still_remaining.append((qi, qj, j_val))

            for qi, qj, j_val in layer:
                schedule.append(
                    {
                        "gate": "J",
                        "qubits": [qi, qj],
                        "time_ns": t_ns,
                        "duration_ns": _GATE_2Q_NS,
                        "coupling_meV": float(j_val),
                        "layer": layer_idx,
                        "description": f"J-gate: exchange q{qi}-q{qj}",
                    }
                )
            t_ns += _GATE_2Q_NS
            layer_idx += 1
            remaining = still_remaining

        # Final layer: parallel A-gate readout
        for q in range(n):
            schedule.append(
                {
                    "gate": "A",
                    "qubits": [q],
                    "time_ns": t_ns,
                    "duration_ns": _GATE_1Q_NS,
                    "layer": layer_idx,
                    "description": f"A-gate: readout qubit {q}",
                }
            )
        t_ns += _GATE_1Q_NS

        return schedule

    def get_constraints(self, n_sites: int) -> dict[str, Any]:
        """Return design constraints for a register of given size."""
        max_distance = self.spacing_nm * (n_sites - 1) if n_sites > 1 else 0
        min_coupling = self._exchange_coupling(max_distance) if n_sites > 1 else _J0_MEV
        nn_coupling = self._exchange_coupling(self.spacing_nm) if n_sites > 1 else _J0_MEV

        return {
            "n_sites": n_sites,
            "topology": self.topology,
            "spacing_nm": self.spacing_nm,
            "max_distance_nm": max_distance,
            "nearest_neighbour_coupling_meV": nn_coupling,
            "weakest_coupling_meV": min_coupling,
            "t2_nuclear_s": _T2_NUCLEAR_S,
            "t2_electron_ms": _T2_ELECTRON_MS,
            "gate_1q_ns": _GATE_1Q_NS,
            "gate_2q_ns": _GATE_2Q_NS,
            "feasible": bool(nn_coupling > 1e-6),
        }

    def __repr__(self) -> str:
        """Return a concise constructor-style mapper description."""
        return (
            f"KaneSiliconMapper(spacing={self.spacing_nm}nm, "
            f"depth={self.depth_nm}nm, topology={self.topology!r})"
        )


__all__ = ["KaneSiliconMapper", "KaneRegisterLayout"]

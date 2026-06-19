# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Non-local spin pool emulator (MPS)

"""Non-local spin pool emulator using Matrix Product States.

Simulates entangled :sup:`31`\\ P nuclear spins in Posner calcium phosphate
molecules (Ca₉(PO₄)₆).  The key observable is ``get_local_atp_efficiency``:
the probability that ATP hydrolysis at a given site is enhanced by non-local
entanglement correlations.

The quantum state is represented by bond-limited MPS tensors.  The
``entanglement_map`` retained in snapshots is a deterministic diagnostic of
where spike-triggered measurements occurred; it is not used as a substitute
for quantum observables.

Mathematical model
------------------

The diagnostic entanglement map e(s) for site s evolves on each spike event
at site s₀:

    influence(s) = exp(-|s - s₀| / ξ)    where ξ = 2.0 (correlation length)
    e(s) ← (1 - α) · e(s) + α · influence(s)
    e(s) ← e(s) / Σ e(s)                 (normalisation)

with α = 0.1 (update rate).

Local ATP hydrolysis efficiency is the two-site singlet probability:

    η(s) = Tr(ρ₁₂ · P_singlet)

where P_singlet projects onto the singlet spin state (Fisher 2015, §4.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SpinCouplingTensor:
    """Two-spin coupling tensor in MHz.

    ``tensor_mhz[a, b]`` multiplies ``S_i^a S_j^b`` for
    ``a,b ∈ {x,y,z}``.  This supports isotropic exchange, anisotropic
    dipolar coupling, and off-diagonal tensor terms without adding hidden
    model constants.
    """

    i: int
    j: int
    tensor_mhz: np.ndarray[Any, Any]


class SpinPoolMPS:
    """Non-local spin storage using true Matrix Product States.

    Simulates entangled :sup:`31`\\ P nuclear spins in Posner molecules.
    Each site corresponds to a phosphorus nuclear spin represented as a
    rank-3 tensor A[α, σ, β] where:
    - α, β are bond indices (dimension ``bond_dim``)
    - σ is the physical index (0 = spin-up, 1 = spin-down)

    The full state |Ψ⟩ = Σ Tr(A¹[σ₁]·A²[σ₂]·…·Aⁿ[σₙ]) |σ₁…σₙ⟩

    Parameters
    ----------
    n_sites : int
        Number of nuclear spin sites.
    bond_dim : int
        Maximum bond dimension (controls entanglement capacity).
    correlation_length : float
        Initialisation parameter for inter-site correlation decay.
    update_rate : float
        Mixing rate α for entanglement map updates on spike events.
    """

    def __init__(
        self,
        n_sites: int = 8,
        bond_dim: int = 16,
        correlation_length: float = 2.0,
        update_rate: float = 0.1,
        seed: int | None = 42,
    ) -> None:
        if n_sites < 1:
            raise ValueError(f"n_sites must be >= 1, got {n_sites}")
        if bond_dim < 1:
            raise ValueError(f"bond_dim must be >= 1, got {bond_dim}")
        if correlation_length <= 0.0:
            raise ValueError(f"correlation_length must be > 0, got {correlation_length}")
        if not 0.0 < update_rate <= 1.0:
            raise ValueError(f"update_rate must be in (0, 1], got {update_rate}")

        self.n_sites = n_sites
        self.bond_dim = bond_dim
        self.correlation_length = correlation_length
        self.update_rate = update_rate
        self._rng = np.random.default_rng(seed)

        # MPS tensors: list of A[α, σ, β] for each site.
        # Initialised to the pure product state |00…0⟩.  Thermal mixed
        # states require density-matrix evolution and are handled in the
        # radical-pair verification path, not by pretending a pure MPS is
        # mixed.
        self.tensors: list[np.ndarray[Any, Any]] = self._init_product_state()

        # Entanglement map (derived quantity, not fundamental)
        self.entanglement_map: np.ndarray[Any, Any] = np.ones(n_sites, dtype=np.float64) / n_sites

        # Two-site singlet projection operator |ψ⁻⟩⟨ψ⁻|
        # |ψ⁻⟩ = (|01⟩ - |10⟩)/√2
        # P_S = [[0,0,0,0],[0,½,-½,0],[0,-½,½,0],[0,0,0,0]] in {|00⟩,|01⟩,|10⟩,|11⟩}
        psi_minus = np.array([0, 1, -1, 0], dtype=np.complex128) / np.sqrt(2)
        self.P_singlet_2site: np.ndarray[Any, Any] = np.outer(psi_minus, psi_minus.conj())
        self._measurement_count = 0

    def _init_product_state(self) -> list[np.ndarray[Any, Any]]:
        """Initialise MPS as the pure product state |00…0⟩."""
        tensors = []
        for _ in range(self.n_sites):
            # A product state has exact Schmidt rank 1 across every bond.
            # ``bond_dim`` is the maximum allowed rank during later evolution,
            # not a reason to allocate zero-filled virtual dimensions here.
            A = np.zeros((1, 2, 1), dtype=np.complex128)
            A[0, 0, 0] = 1.0
            tensors.append(A)
        return tensors

    def to_statevector(self, *, max_sites: int = 16) -> np.ndarray[Any, Any]:
        """Return the exact statevector represented by this MPS.

        This is intended for verification-size systems.  Larger systems must
        be handled with tensor-network algorithms that keep explicit error
        budgets.
        """
        if self.n_sites > max_sites:
            raise ValueError(
                f"Exact statevector export limited to {max_sites} sites, got {self.n_sites}"
            )
        state = self.tensors[0][0, :, :]
        for A in self.tensors[1:]:
            state = np.tensordot(state, A, axes=([-1], [0]))
        vec = state[..., 0].reshape(-1)
        norm = np.linalg.norm(vec)
        if norm == 0.0:
            raise ValueError("MPS state has zero norm")
        normed: np.ndarray[Any, Any] = vec / norm
        return normed

    def set_statevector(
        self,
        statevector: np.ndarray[Any, Any],
        *,
        atol: float = 1e-12,
    ) -> None:
        """Load a statevector into MPS form without silent truncation."""
        vec = np.asarray(statevector, dtype=np.complex128).reshape(-1)
        expected = 1 << self.n_sites
        if vec.size != expected:
            raise ValueError(f"statevector length {vec.size} != 2**n_sites ({expected})")
        norm = np.linalg.norm(vec)
        if norm == 0.0:
            raise ValueError("statevector has zero norm")
        psi = (vec / norm).reshape([2] * self.n_sites)

        tensors: list[np.ndarray[Any, Any]] = []
        left_dim = 1
        work = psi
        for site in range(self.n_sites - 1):
            matrix = work.reshape(left_dim * 2, -1)
            U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
            keep = int(np.sum(atol < S))
            keep = max(1, keep)
            if keep > self.bond_dim:
                discarded = float(np.sum(S[self.bond_dim :] ** 2))
                raise ValueError(
                    f"State requires bond dimension {keep} at bond {site}; "
                    f"configured bond_dim={self.bond_dim}, discarded_norm={discarded:.3e}"
                )
            U = U[:, :keep]
            S = S[:keep]
            Vh = Vh[:keep, :]
            tensors.append(U.reshape(left_dim, 2, keep))
            work = np.diag(S) @ Vh
            left_dim = keep
        tensors.append(work.reshape(left_dim, 2, 1))
        self.tensors = tensors
        self._update_entanglement_map()

    @staticmethod
    def _full_spin_operator(n_sites: int, site: int, component: int) -> np.ndarray[Any, Any]:
        matrices = [
            np.array([[0, 1], [1, 0]], dtype=np.complex128) / 2.0,
            np.array([[0, -1j], [1j, 0]], dtype=np.complex128) / 2.0,
            np.array([[1, 0], [0, -1]], dtype=np.complex128) / 2.0,
        ]
        op = np.array([[1.0 + 0j]])
        ident = np.eye(2, dtype=np.complex128)
        for idx in range(n_sites):
            op = np.kron(op, matrices[component] if idx == site else ident)
        return op

    def evolve_exact(
        self,
        couplings: list[SpinCouplingTensor],
        time_us: float,
        *,
        max_sites: int = 12,
    ) -> None:
        """Evolve under explicit two-spin coupling tensors.

        The Hamiltonian is built exactly as
        ``2π Σ A_ij^{ab} S_i^a S_j^b`` with coupling tensors in MHz and
        time in µs.  The state is converted back to MPS only if the configured
        ``bond_dim`` can represent it without truncation.
        """
        if time_us < 0.0:
            raise ValueError(f"time_us must be >= 0, got {time_us}")
        if self.n_sites > max_sites:
            raise ValueError(
                f"Exact dense Hamiltonian evolution limited to {max_sites} sites, got {self.n_sites}"
            )
        dim = 1 << self.n_sites
        H = np.zeros((dim, dim), dtype=np.complex128)
        ops = [
            [self._full_spin_operator(self.n_sites, site, comp) for comp in range(3)]
            for site in range(self.n_sites)
        ]
        for coupling in couplings:
            if not 0 <= coupling.i < self.n_sites or not 0 <= coupling.j < self.n_sites:
                raise IndexError(f"coupling sites {(coupling.i, coupling.j)} out of range")
            tensor = np.asarray(coupling.tensor_mhz, dtype=np.float64)
            if tensor.shape != (3, 3):
                raise ValueError(f"coupling tensor must have shape (3, 3), got {tensor.shape}")
            for a in range(3):
                for b in range(3):
                    H += (
                        2.0
                        * np.pi
                        * float(tensor[a, b])
                        * (ops[coupling.i][a] @ ops[coupling.j][b])
                    )

        from scipy.linalg import expm

        psi = self.to_statevector(max_sites=max_sites)
        evolved = expm(-1j * H * time_us) @ psi
        self.set_statevector(evolved)

    def _compute_rdm_single(self, site: int) -> np.ndarray[Any, Any]:
        """Compute single-site reduced density matrix by contracting MPS."""
        n = self.n_sites
        # Contract from left up to site
        L = np.ones((1, 1), dtype=np.float64)
        for i in range(site):
            A = self.tensors[i]
            # Contract L[α,α'] with A[α,σ,β] and A*[α',σ,β']
            L = np.einsum("ab,asc,bsd->cd", L, A, A.conj())

        # Site tensor
        A_s = self.tensors[site]

        # Contract from right
        R = np.ones((1, 1), dtype=np.float64)
        for i in range(n - 1, site, -1):
            A = self.tensors[i]
            R = np.einsum("asc,bsd,cd->ab", A, A.conj(), R)

        # RDM: ρ[σ,σ'] = Σ L[α,α'] A[α,σ,β] A*[α',σ',β'] R[β,β']
        rho: np.ndarray[Any, Any] = np.einsum("ab,asc,bud,cd->su", L, A_s, A_s.conj(), R)
        # Normalise
        tr = np.trace(rho)
        if abs(tr) > 0:
            rho /= tr
        return rho

    def _compute_rdm_two_site(self, site: int) -> np.ndarray[Any, Any]:
        """Compute two-site reduced density matrix for sites (site, site+1).

        Returns a 4×4 Hermitian matrix in the computational basis
        {|00⟩, |01⟩, |10⟩, |11⟩}.

        The two-site RDM is needed for computing genuine singlet
        probability: Tr(ρ₁₂ · |ψ⁻⟩⟨ψ⁻|).
        """
        n = self.n_sites
        if site < 0 or site >= n - 1:
            raise IndexError(f"Two-site RDM requires 0 <= site < {n - 1}, got {site}")

        # Contract from left up to site
        L = np.ones((1, 1), dtype=np.complex128)
        for i in range(site):
            A = self.tensors[i]
            L = np.einsum("ab,asc,bsd->cd", L, A, A.conj())

        # Two site tensors
        A_i = self.tensors[site]  # (d_L, 2, d_mid)
        A_j = self.tensors[site + 1]  # (d_mid, 2, d_R)

        # Contract from right
        R = np.ones((1, 1), dtype=np.complex128)
        for i in range(n - 1, site + 1, -1):
            A = self.tensors[i]
            R = np.einsum("asc,bsd,cd->ab", A, A.conj(), R)

        # ρ[σ₁,σ₂,σ₁',σ₂'] = Σ L[α,α'] A_i[α,σ₁,γ] A_j[γ,σ₂,β]
        #                              A_i*[α',σ₁',γ'] A_j*[γ',σ₂',β'] R[β,β']
        rho_4 = np.einsum("ab,asc,cud,bve,ewf,df->suvw", L, A_i, A_j, A_i.conj(), A_j.conj(), R)
        # Reshape to 4×4: index = σ₁*2 + σ₂
        rho: np.ndarray[Any, Any] = rho_4.reshape(4, 4)
        tr = np.trace(rho)
        if abs(tr) > 0:
            rho /= tr
        return rho

    def _compute_entanglement_entropy(self, site: int) -> float:
        """Compute von Neumann entropy of bipartition at site."""
        rdm = self._compute_rdm_single(site)
        eigvals = np.linalg.eigvalsh(rdm.real)  # RDM is Hermitian → real eigenvalues
        eigvals = eigvals[eigvals > 1e-15]
        return float(-np.sum(eigvals * np.log2(eigvals)))

    def _update_entanglement_map(self) -> None:
        """Recompute entanglement map from MPS bond entropies."""
        for i in range(self.n_sites):
            self.entanglement_map[i] = max(self._compute_entanglement_entropy(i), 1e-10)
        total = np.sum(self.entanglement_map)
        if total > 0:
            self.entanglement_map /= total

    def apply_measurement(self, site_idx: int, intensity: float = 1.0) -> None:
        """Apply a Born-rule projective measurement at one spin site.

        This method performs measurement only.  It does not inject recovery
        rotations or distance-kernel propagation.  Physical spin evolution
        must be supplied explicitly through ``evolve_exact()``.
        """
        if not 0 <= site_idx < self.n_sites:
            raise IndexError(f"site_idx {site_idx} out of range for {self.n_sites} sites")
        if intensity < 0.0:
            raise ValueError(f"intensity must be >= 0, got {intensity}")

        A = self.tensors[site_idx]

        # Measurement: compute probabilities for σ=0,1
        rdm = self._compute_rdm_single(site_idx)
        p0 = max(rdm[0, 0].real, 0.0)
        p1 = max(rdm[1, 1].real, 0.0)
        ptot = p0 + p1
        if ptot > 0:
            p0 /= ptot

        # Born-rule outcome.  The RNG can be seeded through __init__ for
        # reproducible verification runs.
        outcome = 0 if self._rng.random() < p0 else 1

        # Project: zero out the other spin component, renormalise
        projected = A.copy()
        projected[:, 1 - outcome, :] = 0.0
        norm = np.sqrt(np.sum(np.abs(projected) ** 2))
        if norm > 0:
            projected /= norm
        self.tensors[site_idx] = projected

        # Update diagnostic map via the measurement-location kernel.  This is
        # metadata for dashboards and state snapshots, not a physics term.
        distances = np.abs(np.arange(self.n_sites, dtype=np.float64) - site_idx)
        influence = np.exp(-distances / self.correlation_length) * intensity
        alpha = self.update_rate
        self.entanglement_map = (1.0 - alpha) * self.entanglement_map + alpha * influence
        total = np.sum(self.entanglement_map)
        if total > 0.0:
            self.entanglement_map /= total

        self._measurement_count += 1

    def _apply_adjacent_unitary(self, i: int, unitary: np.ndarray[Any, Any]) -> None:
        """Apply a two-site unitary to adjacent sites ``i`` and ``i + 1``."""
        j = i + 1
        Ai = self.tensors[i]  # (d_L, 2, d_mid)
        Aj = self.tensors[j]  # (d_mid, 2, d_R)

        # Contract into two-site tensor: Θ[α,σ_i,σ_j,β]
        theta = np.einsum("asc,cud->asud", Ai, Aj)
        d_L, _, _, d_R = theta.shape

        # Apply: reshape Θ to (d_L, 4, d_R), apply U, reshape back
        theta_flat = theta.reshape(d_L, 4, d_R)
        theta_flat = np.einsum("ab,cbd->cad", unitary, theta_flat)
        theta = theta_flat.reshape(d_L, 2, 2, d_R)

        # SVD split: reshape to (d_L*2, 2*d_R) matrix
        M = theta.reshape(d_L * 2, 2 * d_R)
        U_svd, S, Vh = np.linalg.svd(M, full_matrices=False)

        # Truncate to bond_dim
        chi = min(len(S), self.bond_dim)
        U_svd = U_svd[:, :chi]
        S = S[:chi]
        Vh = Vh[:chi, :]

        # Absorb singular values into U (left-canonical)
        U_svd = U_svd @ np.diag(S)

        # Reshape back to MPS tensors
        self.tensors[i] = U_svd.reshape(d_L, 2, chi)
        self.tensors[j] = Vh.reshape(chi, 2, d_R)

    def _swap_adjacent(self, i: int) -> None:
        """Apply an exact adjacent SWAP gate inside the MPS."""
        swap = np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            dtype=np.complex128,
        )
        self._apply_adjacent_unitary(i, swap)

    def _apply_heisenberg_between(self, i: int, j: int, coupling: float) -> None:
        """Apply the Heisenberg gate to arbitrary sites via an exact SWAP network."""
        if i == j:
            return
        if i > j:
            i, j = j, i
        if j == i + 1:
            self._apply_tebd_gate(i, j, coupling)
            return

        for k in range(i, j - 1):
            self._swap_adjacent(k)
        self._apply_tebd_gate(j - 1, j, coupling)
        for k in range(j - 2, i - 1, -1):
            self._swap_adjacent(k)

    def _apply_tebd_gate(self, i: int, j: int, coupling: float) -> None:
        """Apply a two-site TEBD gate between adjacent sites i and j.

        Implements the FULL isotropic Heisenberg coupling:
          H = J · (σx⊗σx + σy⊗σy + σz⊗σz)
          U = exp(-i·θ·H)

        The 4×4 unitary is computed via exact matrix exponentiation.
        This is NOT an Ising-only (ZZ) approximation — all three
        exchange channels are included, which is essential for:
        - Correct singlet-triplet splitting (singlet gets -3J, triplets +J)
        - Proper spin-flip dynamics (XX+YY enables ΔSz=±1 transitions)
        - Physically accurate entanglement generation

        Steps:
        1. Contract A[i] and A[j] into a two-site tensor Θ[α,σ_i,σ_j,β]
        2. Apply the 4×4 Heisenberg unitary
        3. SVD to split back into A[i]' and A[j]', truncating bond dim.
        """
        assert j == i + 1, "TEBD requires adjacent sites"
        # Build 4×4 Heisenberg unitary: exp(-iθ·(XX + YY + ZZ))
        #
        # H_Heis = σx⊗σx + σy⊗σy + σz⊗σz
        #        = 2·SWAP - I  (in terms of the SWAP operator)
        #
        # In the computational basis {|00⟩, |01⟩, |10⟩, |11⟩}:
        #   H = [[1,0,0,0], [0,-1,2,0], [0,2,-1,0], [0,0,0,1]]
        #
        # Eigenvalues: +1 (triplet, 3-fold) and -3 (singlet, 1-fold)
        # The singlet |ψ⁻⟩ = (|01⟩-|10⟩)/√2 evolves as e^{+3iθ}
        # The triplets evolve as e^{-iθ}
        #
        # Coupling is dimensionless in this event-driven spin-pool model.
        # Physical Posner verification uses tools/verify_ibm_heron.py with
        # explicit hyperfine tensors and time units.
        angle = coupling * self.update_rate

        # Pauli matrices
        sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        I2 = np.eye(2, dtype=np.complex128)

        # Full Heisenberg Hamiltonian: XX + YY + ZZ
        H = np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz)

        # Exact matrix exponentiation
        from scipy.linalg import expm

        U = expm(-1j * angle * H)

        self._apply_adjacent_unitary(i, U)

    def get_local_atp_efficiency(self, site_idx: int) -> float:
        """Return ATP hydrolysis probability at a given site.

        Computes the quantum observable directly:
        ``Tr(ρ₁₂ · |ψ⁻⟩⟨ψ⁻|)`` on the adjacent two-site reduced density
        matrix.  The diagnostic entanglement map is intentionally excluded
        so this method cannot manufacture ATP gain from a classical proxy.
        """
        if not 0 <= site_idx < self.n_sites:
            raise IndexError(f"site_idx {site_idx} out of range for {self.n_sites} sites")
        # Choose pair: prefer (site, site+1); for last site use (site-1, site)
        if site_idx < self.n_sites - 1:
            pair_site = site_idx
        else:
            pair_site = max(0, site_idx - 1)

        rdm2 = self._compute_rdm_two_site(pair_site)
        singlet_prob = float(np.real(np.trace(rdm2 @ self.P_singlet_2site)))

        return float(np.clip(singlet_prob, 0.0, 1.0))

    @property
    def rho(self) -> np.ndarray[Any, Any]:
        """Return site-0 reduced density matrix (backward compat)."""
        return self._compute_rdm_single(0)

    def get_status(self) -> dict[str, Any]:
        """Return summary status for telemetry and visualisation."""
        return {
            "n_sites": self.n_sites,
            "bond_dim": self.bond_dim,
            "avg_entanglement": float(np.mean(self.entanglement_map)),
            "max_entanglement": float(np.max(self.entanglement_map)),
            "min_entanglement": float(np.min(self.entanglement_map)),
            "measurement_count": self._measurement_count,
            "coherence_status": "stable",
        }

    def get_state(self) -> dict[str, Any]:
        """Return full internal state for checkpointing."""
        return {
            "n_sites": self.n_sites,
            "bond_dim": self.bond_dim,
            "correlation_length": self.correlation_length,
            "update_rate": self.update_rate,
            "tensors": [t.tolist() for t in self.tensors],
            "entanglement_map": self.entanglement_map.tolist(),
            "measurement_count": self._measurement_count,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore internal state from a checkpoint dictionary."""
        if "tensors" in state:
            self.tensors = [np.asarray(t, dtype=np.complex128) for t in state["tensors"]]
        if "entanglement_map" in state:
            self.entanglement_map = np.asarray(state["entanglement_map"], dtype=np.float64)
        else:
            self._update_entanglement_map()
        self._measurement_count = int(state.get("measurement_count", 0))

    def reset(self) -> None:
        """Reset to product state."""
        self.tensors = self._init_product_state()
        self.entanglement_map = np.ones(self.n_sites, dtype=np.float64) / self.n_sites
        self._measurement_count = 0

    def to_scpn_payload(self) -> dict[str, Any]:
        """Produce metadata compatible with SCPNDatastream format."""
        return {
            "quantum_cognition_spin_pool": {
                "n_sites": self.n_sites,
                "entanglement_map": self.entanglement_map.tolist(),
                "atp_efficiencies": [self.get_local_atp_efficiency(i) for i in range(self.n_sites)],
            },
        }

    def __repr__(self) -> str:
        avg_e = float(np.mean(self.entanglement_map))
        return (
            f"SpinPoolMPS(n_sites={self.n_sites}, bond_dim={self.bond_dim}, "
            f"avg_entanglement={avg_e:.4f}, measurements={self._measurement_count})"
        )


__all__ = ["SpinCouplingTensor", "SpinPoolMPS"]

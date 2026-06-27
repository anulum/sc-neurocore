# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Radical Pair Mechanism model

r"""Radical Pair Mechanism (RPM) for quantum-metabolic coupling.

Computes singlet–triplet interconversion by evolving the full
electron+nuclear spin density matrix under an explicit radical-pair
Hamiltonian.  The default is an exact one-nucleus isotropic RPM used for
software tests.  Posner-molecule calculations must provide measured or
DFT-derived hyperfine tensors; this module does not infer molecular tensors.

Background
----------
When ATP is hydrolysed, the phosphorus radical pair passes through a
superposition of singlet (↑↓) and triplet (↑↑) electron-spin states.
The singlet channel permits the reaction to proceed (ATP → ADP + Pᵢ),
while the triplet channel blocks it.  An external or local magnetic
field shifts the singlet–triplet mixing rate, providing a mechanism
for quantum control of metabolic energy release.

The Hamiltonian follows the standard RPM form used in spin chemistry.

Mathematical model
------------------
.. math::

    H = \omega_e(S_{1z}+S_{2z})
        + 2\pi J\,\mathbf{S}_1\cdot\mathbf{S}_2
        + 2\pi\sum_k \mathbf{S}_{r(k)} A_k \mathbf{I}_k

    \Phi_S =
      \frac{\int_0^\tau k e^{-kt}\operatorname{Tr}
      [P_S e^{-iHt}\rho_0 e^{iHt}]\,dt}{1-e^{-k\tau}}

Hyperfine tensors and exchange are in MHz, time is in µs, and ``H`` is
in rad/µs.

References
----------
- Schulten, K. & Wolynes, P. G. "Semiclassical description of electron
  spin motion in radicals." *J. Chem. Phys.* 68 (1978).
- Hore, P. J. & Mouritsen, H. "The radical-pair mechanism of
  magnetoreception." *Ann. Rev. Biophys.* 45 (2016).
- Fisher, M. P. A. "Quantum cognition." *Annals of Physics* 362 (2015).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Physical constants
_GAMMA_E = 1.760_859_630_23e11  # electron gyromagnetic ratio [rad/(s·T)]
_GAMMA_E_RAD_PER_US_T = _GAMMA_E * 1e-6
_RAD_PER_US_PER_MHZ = 2.0 * np.pi


@dataclass
class RadicalPairParams:
    """Parameters for a radical pair system.

    Attributes
    ----------
    hyperfine_a : float
        Isotropic hyperfine coupling constant in MHz for the exact default
        one-nucleus model.  Posner-specific calculations must pass explicit
        tensors via ``hyperfine_tensors_1`` and/or ``hyperfine_tensors_2``.
    exchange_j : float
        Exchange coupling in MHz.
        Typical: 0–10 MHz for separated radicals in Posner molecules.
    recombination_rate : float
        Radical pair recombination rate in µs⁻¹.
        Typical: 0.01–1.0 µs⁻¹.
    lifetime_us : float
        Radical pair lifetime in µs.
        Posner molecules: ~1–1000 µs (protected by Ca₉(PO₄)₆ cage).
    hyperfine_tensors_1, hyperfine_tensors_2 : list[np.ndarray[Any, Any]]
        3×3 hyperfine tensors in MHz for nuclei coupled to electron 1 and
        electron 2 respectively.
    quadrature_order : int
        Gauss-Legendre points for finite-lifetime recombination integration.
    """

    hyperfine_a: float = 10.0  # MHz
    exchange_j: float = 1.0  # MHz
    recombination_rate: float = 0.1  # µs⁻¹
    lifetime_us: float = 100.0  # µs
    hyperfine_tensors_1: list[np.ndarray[Any, Any]] = field(default_factory=list)
    hyperfine_tensors_2: list[np.ndarray[Any, Any]] = field(default_factory=list)
    quadrature_order: int = 64


class RadicalPairModel:
    """Density-matrix radical pair mechanism for ATP hydrolysis gating.

    Computes singlet yield (probability of productive ATP hydrolysis)
    as a function of local magnetic field and radical pair parameters.

    Parameters
    ----------
    params : RadicalPairParams, optional
        RPM parameters.  Defaults are an exact one-nucleus isotropic RPM,
        not a Posner molecule parameterization.
    """

    def __init__(self, params: RadicalPairParams | None = None) -> None:
        self.params = params or RadicalPairParams()
        if self.params.quadrature_order < 2:
            raise ValueError("quadrature_order must be >= 2")

    @staticmethod
    def _isotropic_tensor(a_mhz: float) -> np.ndarray[Any, Any]:
        return np.eye(3, dtype=np.float64) * float(a_mhz)

    @classmethod
    def from_hyperfine_tensors(
        cls,
        *,
        tensors_1: list[np.ndarray[Any, Any]],
        tensors_2: list[np.ndarray[Any, Any]],
        exchange_j: float,
        recombination_rate: float,
        lifetime_us: float,
        quadrature_order: int = 64,
    ) -> "RadicalPairModel":
        """Construct an exact RPM model from explicit 3×3 hyperfine tensors."""
        return cls(
            RadicalPairParams(
                hyperfine_a=0.0,
                exchange_j=exchange_j,
                recombination_rate=recombination_rate,
                lifetime_us=lifetime_us,
                hyperfine_tensors_1=[np.asarray(t, dtype=np.float64) for t in tensors_1],
                hyperfine_tensors_2=[np.asarray(t, dtype=np.float64) for t in tensors_2],
                quadrature_order=quadrature_order,
            )
        )

    def _validated_tensors(self) -> tuple[list[np.ndarray[Any, Any]], list[np.ndarray[Any, Any]]]:
        p = self.params
        tensors_1 = [np.asarray(t, dtype=np.float64) for t in p.hyperfine_tensors_1]
        tensors_2 = [np.asarray(t, dtype=np.float64) for t in p.hyperfine_tensors_2]
        if not tensors_1 and not tensors_2:
            tensors_1 = [self._isotropic_tensor(p.hyperfine_a)]

        for label, tensors in (
            ("hyperfine_tensors_1", tensors_1),
            ("hyperfine_tensors_2", tensors_2),
        ):
            for idx, tensor in enumerate(tensors):
                if tensor.shape != (3, 3):
                    raise ValueError(f"{label}[{idx}] must have shape (3, 3), got {tensor.shape}")
        return tensors_1, tensors_2

    @staticmethod
    def _spin_operator(n_spins: int, target: int, component: str) -> np.ndarray[Any, Any]:
        matrices = {
            "x": np.array([[0, 1], [1, 0]], dtype=np.complex128) / 2.0,
            "y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128) / 2.0,
            "z": np.array([[1, 0], [0, -1]], dtype=np.complex128) / 2.0,
        }
        op = np.array([[1.0 + 0j]])
        ident = np.eye(2, dtype=np.complex128)
        for idx in range(n_spins):
            op = np.kron(op, matrices[component] if idx == target else ident)
        return op

    @staticmethod
    def _singlet_density_with_nuclear_bath(
        n_nuclei: int,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        singlet = np.array([0, 1, -1, 0], dtype=np.complex128) / np.sqrt(2.0)
        rho_e = np.outer(singlet, singlet.conj())
        if n_nuclei == 0:
            return rho_e, rho_e.copy()

        bath_dim = 1 << n_nuclei
        bath = np.eye(bath_dim, dtype=np.complex128) / bath_dim
        return (
            np.kron(rho_e, bath),
            np.kron(rho_e, np.eye(bath_dim, dtype=np.complex128)),
        )

    def _hamiltonian(self, b_local: float) -> tuple[np.ndarray[Any, Any], int]:
        p = self.params
        tensors_1, tensors_2 = self._validated_tensors()
        n_nuclei = len(tensors_1) + len(tensors_2)
        n_spins = 2 + n_nuclei
        if n_spins > 10:
            raise ValueError(
                f"Exact dense RPM evolution supports up to 8 nuclei (10 spins), got {n_nuclei}. "
                "Use a tensor-network solver for larger nuclear baths."
            )

        sx = [self._spin_operator(n_spins, i, "x") for i in range(n_spins)]
        sy = [self._spin_operator(n_spins, i, "y") for i in range(n_spins)]
        sz = [self._spin_operator(n_spins, i, "z") for i in range(n_spins)]
        comps = [sx, sy, sz]

        dim = 1 << n_spins
        H = np.zeros((dim, dim), dtype=np.complex128)
        H += _GAMMA_E_RAD_PER_US_T * float(b_local) * (sz[0] + sz[1])
        H += (
            _RAD_PER_US_PER_MHZ
            * float(p.exchange_j)
            * (sx[0] @ sx[1] + sy[0] @ sy[1] + sz[0] @ sz[1])
        )

        nuc_idx = 2
        for electron_idx, tensors in ((0, tensors_1), (1, tensors_2)):
            for tensor in tensors:
                for a in range(3):
                    for b in range(3):
                        H += (
                            _RAD_PER_US_PER_MHZ
                            * float(tensor[a, b])
                            * (comps[a][electron_idx] @ comps[b][nuc_idx])
                        )
                nuc_idx += 1
        return H, n_nuclei

    def singlet_yield(self, b_local: float = 0.0) -> float:
        """Compute singlet yield Φ_S for the current parameters.

        Parameters
        ----------
        b_local : float
            Local magnetic field in Tesla.  Earth's field ≈ 50 µT.
            Neural spike-generated fields ≈ 0.1–1 nT.

        Returns
        -------
        float
            Singlet yield in [0, 1].  Higher values mean more
            efficient ATP hydrolysis (singlet channel open).
        """
        p = self.params
        if p.recombination_rate <= 0.0:
            raise ValueError("recombination_rate must be > 0")
        if p.lifetime_us <= 0.0:
            raise ValueError("lifetime_us must be > 0")

        H, n_nuclei = self._hamiltonian(b_local)
        rho0, p_s = self._singlet_density_with_nuclear_bath(n_nuclei)

        from scipy.linalg import expm

        nodes, weights = np.polynomial.legendre.leggauss(p.quadrature_order)
        times = 0.5 * p.lifetime_us * (nodes + 1.0)
        scaled_weights = 0.5 * p.lifetime_us * weights
        norm = 1.0 - np.exp(-p.recombination_rate * p.lifetime_us)

        numerator = 0.0
        for t_us, weight in zip(times, scaled_weights, strict=True):
            U = expm(-1j * H * t_us)
            rho_t = U @ rho0 @ U.conj().T
            p_s_t = float(np.real(np.trace(p_s @ rho_t)))
            numerator += (
                weight * p.recombination_rate * np.exp(-p.recombination_rate * t_us) * p_s_t
            )

        return float(np.clip(numerator / norm, 0.0, 1.0))

    def singlet_yield_field_sweep(self, b_range: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Compute singlet yield over a range of magnetic fields.

        Parameters
        ----------
        b_range : np.ndarray[Any, Any]
            Array of magnetic field values in Tesla.

        Returns
        -------
        np.ndarray[Any, Any]
            Singlet yield at each field value.
        """
        return np.array([self.singlet_yield(b) for b in b_range])

    def atp_efficiency(
        self,
        b_local: float = 0.0,
        entanglement_boost: float = 0.0,
    ) -> float:
        """Compute ATP hydrolysis efficiency from singlet yield.

        Parameters
        ----------
        b_local : float
            Local magnetic field in Tesla.
        entanglement_boost : float
            Unsupported legacy argument.  Non-zero values raise because a
            classical boost is not a radical-pair Hamiltonian parameter.

        Returns
        -------
        float
            Singlet-yield-derived ATP efficiency in [0, 1].
        """
        if entanglement_boost != 0.0:
            raise ValueError(
                "entanglement_boost is not a physical RPM parameter; provide "
                "explicit hyperfine tensors or combine observables upstream."
            )
        return self.singlet_yield(b_local)

    def get_state(self) -> dict[str, Any]:
        """Return parameters as a dict for serialisation."""
        p = self.params
        return {
            "hyperfine_a_mhz": p.hyperfine_a,
            "exchange_j_mhz": p.exchange_j,
            "recombination_rate_us_inv": p.recombination_rate,
            "lifetime_us": p.lifetime_us,
            "n_hyperfine_tensors_1": len(p.hyperfine_tensors_1),
            "n_hyperfine_tensors_2": len(p.hyperfine_tensors_2),
            "quadrature_order": p.quadrature_order,
        }

    def __repr__(self) -> str:
        """Return a concise representation of the active RPM parameters."""
        p = self.params
        return (
            f"RadicalPairModel(J={p.exchange_j}MHz, k={p.recombination_rate}µs⁻¹, "
            f"τ={p.lifetime_us}µs, nuclei="
            f"{len(p.hyperfine_tensors_1) + len(p.hyperfine_tensors_2) or 1})"
        )


__all__ = ["RadicalPairModel", "RadicalPairParams"]

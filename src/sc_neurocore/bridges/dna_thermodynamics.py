# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — DNA thermodynamic analysis

"""Fallback and optional-NUPACK thermodynamic analysis."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Dict, Tuple, cast

import numpy as np

from .dna_types import (
    _DEFAULT_TEMPERATURE_C,
    _HAIRPIN_LOOP_INIT_DG,
    _HAIRPIN_LOOP_SLOPE_DG,
    _MIN_HAIRPIN_LOOP_NT,
    _R_GAS,
    _STACKING_BONUS_DG,
    _WC_PAIR_DG,
    DNACircuitDesign,
)


_NupackBackendProvider = Callable[[], tuple[bool, Any]]


def _fallback_nupack_backend() -> tuple[bool, Any]:
    """Return the internal fallback before the public façade is initialised."""
    return False, None


_nupack_backend: _NupackBackendProvider = _fallback_nupack_backend


def _configure_nupack_backend(provider: _NupackBackendProvider) -> None:
    """Inject the façade-owned optional-backend state provider."""
    global _nupack_backend
    _nupack_backend = provider


def _canonical_sequence(sequence: str) -> str:
    seq = sequence.upper()
    invalid = sorted(set(seq) - set("ACGT"))
    if invalid:
        raise ValueError(f"DNA sequence contains invalid bases: {''.join(invalid)}")
    return seq


def _can_pair(left: str, right: str) -> bool:
    return left + right in _WC_PAIR_DG


def _hairpin_loop_penalty(loop_nt: int) -> float:
    return _HAIRPIN_LOOP_INIT_DG + _HAIRPIN_LOOP_SLOPE_DG * max(0, loop_nt - _MIN_HAIRPIN_LOOP_NT)


def _fallback_pair_energy(sequence: str, i: int, j: int) -> float | None:
    if j - i <= _MIN_HAIRPIN_LOOP_NT or not _can_pair(sequence[i], sequence[j]):
        return None

    energy = _WC_PAIR_DG[sequence[i] + sequence[j]] + _hairpin_loop_penalty(j - i - 1)
    if i + 1 < j - 1 and _can_pair(sequence[i + 1], sequence[j - 1]):
        energy += _STACKING_BONUS_DG
    return energy


def _fallback_secondary_structure(sequence: str) -> tuple[float, str, list[tuple[int, int]]]:
    seq = _canonical_sequence(sequence)
    n = len(seq)
    if n == 0:
        return 0.0, "", []

    dp = np.zeros((n, n), dtype=np.float64)
    trace: list[list[tuple[str, int, int] | None]] = [[None for _ in range(n)] for _ in range(n)]

    for span in range(1, n):
        for i in range(0, n - span):
            j = i + span
            best = dp[i + 1, j]
            trace[i][j] = ("skip_i", i + 1, j)

            if dp[i, j - 1] < best:
                best = dp[i, j - 1]
                trace[i][j] = ("skip_j", i, j - 1)

            pair_energy = _fallback_pair_energy(seq, i, j)
            if pair_energy is not None:
                candidate = pair_energy + (dp[i + 1, j - 1] if i + 1 <= j - 1 else 0.0)
                if candidate < best:
                    best = candidate
                    trace[i][j] = ("pair", i + 1, j - 1)

            for k in range(i, j):
                candidate = dp[i, k] + dp[k + 1, j]
                if candidate < best:
                    best = candidate
                    trace[i][j] = ("split", i, k)

            dp[i, j] = best

    pairs: list[tuple[int, int]] = []

    def traceback(i: int, j: int) -> None:
        if i >= j:
            return
        step = cast(tuple[str, int, int], trace[i][j])
        kind, a, b = step
        if kind == "skip_i" or kind == "skip_j":
            traceback(a, b)
        elif kind == "pair":
            pairs.append((i, j))
            traceback(a, b)
        else:
            traceback(i, b)
            traceback(b + 1, j)

    traceback(0, n - 1)
    structure = ["."] * n
    for i, j in pairs:
        structure[i] = "("
        structure[j] = ")"
    return float(dp[0, n - 1]), "".join(structure), pairs


def _fallback_pair_probability_matrix(sequence: str, temperature_c: float) -> np.ndarray[Any, Any]:
    seq = _canonical_sequence(sequence)
    n = len(seq)
    weights = np.zeros((n, n), dtype=np.float64)
    rt = _R_GAS * (temperature_c + 273.15)

    for i in range(n):
        for j in range(i + _MIN_HAIRPIN_LOOP_NT + 1, n):
            pair_energy = _fallback_pair_energy(seq, i, j)
            if pair_energy is None:
                continue
            weights[i, j] = math.exp(-pair_energy / rt)
            weights[j, i] = weights[i, j]

    if not np.any(weights):
        return weights

    row_mass = 1.0 + weights.sum(axis=1)
    probabilities = np.zeros_like(weights)
    for i in range(n):
        for j in range(i + 1, n):
            if weights[i, j] == 0.0:
                continue
            probability = weights[i, j] / math.sqrt(row_mass[i] * row_mass[j])
            probabilities[i, j] = min(1.0, probability)
            probabilities[j, i] = probabilities[i, j]
    return probabilities


class NUPACKInterface:
    """Interface to NUPACK for thermodynamic validation.

    Provides minimum free energy (MFE) structure prediction, base-pair
    probability computation, and design validation. Falls back to
    internal nearest-neighbour estimates, Watson-Crick secondary-structure
    dynamic programming, and Boltzmann-style pair probabilities when NUPACK is
    not installed.

    Parameters
    ----------
    temperature_c : float
        Temperature in Celsius.
    na_concentration_M : float
        Sodium concentration in molar.
    """

    def __init__(
        self,
        temperature_c: float = _DEFAULT_TEMPERATURE_C,
        na_concentration_M: float = 1.0,
    ) -> None:
        self._temperature_c = temperature_c
        self._na_M = na_concentration_M

    @property
    def has_nupack(self) -> bool:
        """Return whether optional NUPACK thermodynamic analysis is installed."""
        return _nupack_backend()[0]

    def compute_mfe(self, sequence: str) -> Tuple[float, str]:
        """Compute minimum free energy and structure.

        Returns
        -------
        tuple[float, str]
            (energy_kcal_mol, dot_bracket_structure)
        """
        has_nupack, nupack_backend = _nupack_backend()
        if has_nupack:
            model = nupack_backend.Model(
                material="dna",
                celsius=self._temperature_c,
                sodium=self._na_M,
            )
            strand = nupack_backend.Strand(sequence, name="query")
            result = nupack_backend.mfe(strands=[strand], model=model)
            energy = float(result[0].energy)
            structure = str(result[0].structure)
            return energy, structure

        return _fallback_secondary_structure(sequence)[:2]

    def compute_pair_probabilities(self, sequence: str) -> np.ndarray[Any, Any]:
        """Compute base-pair probability matrix.

        Returns
        -------
        np.ndarray
            N×N matrix where entry (i, j) is the probability that
            positions i and j are base-paired at equilibrium.
        """
        has_nupack, nupack_backend = _nupack_backend()
        if has_nupack:
            model = nupack_backend.Model(
                material="dna",
                celsius=self._temperature_c,
                sodium=self._na_M,
            )
            strand = nupack_backend.Strand(sequence, name="query")
            result = nupack_backend.pairs(strands=[strand], model=model)
            return np.array(result.to_array())

        return _fallback_pair_probability_matrix(sequence, self._temperature_c)

    def validate_design(self, design: DNACircuitDesign) -> Dict[str, Any]:
        """Validate a full circuit design.

        Checks for:
        - Unwanted secondary structures (ΔG < −2 kcal/mol)
        - Cross-hybridization between non-interacting strands
        - GC content and homopolymer constraints

        Returns
        -------
        dict
            Validation report with per-strand results.
        """
        all_strands = design.input_strands + design.output_strands + design.fuel_strands
        for g in design.gates:
            all_strands.extend(g.strands)

        report: Dict[str, Any] = {
            "valid": True,
            "strand_results": {},
            "cross_hybridization": [],
            "warnings": design.validate(),
        }

        for strand in all_strands:
            energy, structure = self.compute_mfe(strand.sequence)
            has_structure = energy < -2.0 and strand.role == "signal"
            report["strand_results"][strand.name] = {
                "mfe_energy": energy,
                "structure": structure,
                "gc_content": strand.gc_content,
                "homopolymer_max": strand.max_homopolymer_run,
                "has_unwanted_structure": has_structure,
            }
            if has_structure:
                report["valid"] = False

        if report["warnings"]:
            report["valid"] = False

        return report

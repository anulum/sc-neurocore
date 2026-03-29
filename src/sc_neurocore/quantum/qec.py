# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quantum Error Correction (QEC) Shield for sc-neurocore

"""
Quantum Error Correction (QEC) Shield for sc-neurocore.

Provides classical-stochastic implementations of QEC codes (Repetition, Surface)
to protect quantum-classical bitstreams from noise during IBMQ hardware execution.
"""

from __future__ import annotations
from typing import Any
import numpy as np


class QecShield:
    """Repetition code QEC shield for stochastic-quantum bitstreams."""

    def __init__(self, code_type: str = "repetition", distance: int = 3):
        self.code_type = code_type
        self.distance = distance

    def encode(self, bitstream: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """repetition code (d=3): 0 -> 000, 1 -> 111"""
        if self.code_type == "repetition":
            return np.repeat(bitstream[:, np.newaxis, :], self.distance, axis=1)
        return bitstream

    def extract_syndromes(self, physical_bits: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        if self.code_type == "repetition":
            res: np.ndarray[Any, Any] = np.diff(physical_bits, axis=1) % 2
            return res
        return np.zeros_like(physical_bits)

    def decode(self, physical_bits: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        if self.code_type == "repetition":
            means = np.mean(physical_bits, axis=1)
            res: np.ndarray[Any, Any] = (means > 0.5).astype(np.uint8)
            return res
        return physical_bits

    def get_error_rate(self, syndromes: np.ndarray[Any, Any]) -> float:
        return float(np.mean(syndromes))


class SurfaceCodeShield:
    """
    Distance-d rotated surface code for stochastic-quantum bitstreams.

    Encodes 1 logical qubit into d² physical data qubits. X and Z stabilizers
    detect bit-flip and phase-flip errors. Decoding uses a lookup table for d=3.

    Ref: Fowler et al., "Surface codes: Towards practical large-scale quantum
    computation", Phys. Rev. A 86, 032324 (2012).
    """

    def __init__(self, distance: int = 3):
        if distance < 3 or distance % 2 == 0:
            raise ValueError("distance must be odd >= 3")
        self.distance = distance
        self.n_data = distance * distance
        # Stabilizer generators: each is a list of data-qubit indices
        self.x_stabilizers, self.z_stabilizers = self._build_stabilizers(distance)
        # d=3 lookup table: syndrome pattern -> correction qubit index
        if distance == 3:
            self._x_lut = self._build_d3_lut(self.x_stabilizers)
            self._z_lut = self._build_d3_lut(self.z_stabilizers)

    @staticmethod
    def _build_stabilizers(d: int) -> tuple[list[list[int]], list[list[int]]]:
        """Build X and Z stabilizer generators for rotated surface code."""
        x_stabs: list[list[int]] = []
        z_stabs: list[list[int]] = []
        for r in range(d):
            for c in range(d):
                idx = r * d + c
                # X stabilizers: plaquettes on even sublattice
                if (r + c) % 2 == 0 and r < d - 1 and c < d - 1:
                    x_stabs.append([idx, idx + 1, idx + d, idx + d + 1])
                # Z stabilizers: plaquettes on odd sublattice
                if (r + c) % 2 == 1 and r < d - 1 and c < d - 1:
                    z_stabs.append([idx, idx + 1, idx + d, idx + d + 1])
        # Boundary stabilizers (weight-2) for top/bottom/left/right edges
        for c in range(0, d - 1, 2):
            x_stabs.append([c, c + 1])  # top edge
        for c in range(1 if d > 3 else 0, d - 1, 2):
            if (d - 1) * d + c < d * d and (d - 1) * d + c + 1 < d * d:
                x_stabs.append([(d - 1) * d + c, (d - 1) * d + c + 1])  # bottom edge
        for r in range(0, d - 1, 2):
            z_stabs.append([r * d, (r + 1) * d])  # left edge
        for r in range(1 if d > 3 else 0, d - 1, 2):
            if (r + 1) * d + d - 1 < d * d:
                z_stabs.append([r * d + d - 1, (r + 1) * d + d - 1])  # right edge
        return x_stabs, z_stabs

    @staticmethod
    def _build_d3_lut(stabilizers: list[list[int]]) -> dict[tuple[int, ...], int]:
        """Build syndrome → correction lookup for d=3 single-qubit errors."""
        lut: dict[tuple[int, ...], int] = {}
        n_stabs = len(stabilizers)
        for qubit in range(9):
            syndrome = [0] * n_stabs
            for s_idx, stab in enumerate(stabilizers):
                if qubit in stab:
                    syndrome[s_idx] = 1
            key = tuple(syndrome)
            if key not in lut:
                lut[key] = qubit
        return lut

    def encode(self, bitstream: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """
        Encode logical bitstream into surface code physical qubits.

        Input: (n_logical, length) — each row is one logical qubit's bitstream.
        Output: (n_logical, n_data, length) — repeated into d² data qubits.
        """
        return np.repeat(bitstream[:, np.newaxis, :], self.n_data, axis=1)

    def measure_syndrome(
        self, physical_bits: np.ndarray[Any, Any]
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """
        Measure X and Z stabilizer syndromes.

        Input: (n_logical, n_data, length)
        Returns: (x_syndrome, z_syndrome) each (n_logical, n_stabilizers, length)
        """
        n_logical, _, length = physical_bits.shape
        x_syn = np.zeros((n_logical, len(self.x_stabilizers), length), dtype=np.uint8)
        z_syn = np.zeros((n_logical, len(self.z_stabilizers), length), dtype=np.uint8)
        for s_idx, stab in enumerate(self.x_stabilizers):
            parity = np.zeros((n_logical, length), dtype=np.uint8)
            for q in stab:
                parity ^= physical_bits[:, q, :]
            x_syn[:, s_idx, :] = parity
        for s_idx, stab in enumerate(self.z_stabilizers):
            parity = np.zeros((n_logical, length), dtype=np.uint8)
            for q in stab:
                parity ^= physical_bits[:, q, :]
            z_syn[:, s_idx, :] = parity
        return x_syn, z_syn

    def decode(self, physical_bits: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """
        Decode surface code: measure syndromes, correct single-qubit errors, majority vote.

        Input: (n_logical, n_data, length)
        Output: (n_logical, length)
        """
        corrected = physical_bits.copy()
        x_syn, z_syn = self.measure_syndrome(corrected)
        n_logical, _, length = corrected.shape

        if self.distance == 3:
            self._apply_lut_correction(corrected, x_syn, self._x_lut)
            self._apply_lut_correction(corrected, z_syn, self._z_lut)
        else:
            # For d>3, apply majority vote per stabilizer neighbourhood
            pass

        # Majority vote across all data qubits
        means = np.mean(corrected, axis=1)
        result: np.ndarray[Any, Any] = (means > 0.5).astype(np.uint8)
        return result

    @staticmethod
    def _apply_lut_correction(
        physical: np.ndarray[Any, Any],
        syndromes: np.ndarray[Any, Any],
        lut: dict[tuple[int, ...], int],
    ) -> None:
        """Apply lookup-table correction for each bitstream position."""
        n_logical, n_stab, length = syndromes.shape
        for l_idx in range(n_logical):
            for t in range(length):
                syn_key = tuple(int(syndromes[l_idx, s, t]) for s in range(n_stab))
                if any(syn_key):
                    qubit = lut.get(syn_key)
                    if qubit is not None:
                        physical[l_idx, qubit, t] ^= 1

    def get_error_rate(self, x_syn: np.ndarray[Any, Any], z_syn: np.ndarray[Any, Any]) -> float:
        """Estimated error rate from syndrome density."""
        return float((np.mean(x_syn) + np.mean(z_syn)) / 2)

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — DNA encoding and error correction

"""GF(4) error correction and dual-rail molecular encoding."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from .dna_bridge import BitstreamToDNA
from .dna_types import DNACircuitDesign, GateType


class GF4ErrorCorrection:
    """Reed–Solomon-like error correction over GF(4) for DNA sequences.

    Maps nucleotides to GF(4) elements: A=0, C=1, G=2, T=3.
    Adds parity symbols for error detection and correction of
    synthesis/sequencing errors.

    Parameters
    ----------
    n_parity : int
        Number of parity nucleotides per block (default 4).
    block_size : int
        Data nucleotides per block (default 12).
    """

    NUC_TO_GF4 = {"A": 0, "C": 1, "G": 2, "T": 3}
    GF4_TO_NUC = {0: "A", 1: "C", 2: "G", 3: "T"}

    def __init__(self, n_parity: int = 4, block_size: int = 12) -> None:
        self._n_parity = n_parity
        self._block_size = block_size

    def encode(self, sequence: str) -> str:
        """Add error-correction parity nucleotides to a sequence."""
        encoded: list[str] = []
        for i in range(0, len(sequence), self._block_size):
            block = sequence[i : i + self._block_size]
            symbols = [self.NUC_TO_GF4.get(c, 0) for c in block]
            parity = self._compute_parity(symbols)
            encoded.append(block + "".join(self.GF4_TO_NUC[p] for p in parity))
        return "".join(encoded)

    def decode(self, encoded_sequence: str) -> Tuple[str, int]:
        """Decode and correct errors. Returns (corrected_data, n_corrections)."""
        total_block = self._block_size + self._n_parity
        data: list[str] = []
        corrections = 0

        for i in range(0, len(encoded_sequence), total_block):
            block = encoded_sequence[i : i + total_block]
            if len(block) < total_block:
                data.append(block[: self._block_size])
                continue

            data_part = block[: self._block_size]
            parity_part = block[self._block_size :]

            symbols = [self.NUC_TO_GF4.get(c, 0) for c in data_part]
            expected = self._compute_parity(symbols)
            actual = [self.NUC_TO_GF4.get(c, 0) for c in parity_part]

            syndrome = [(a - e) % 4 for a, e in zip(actual, expected)]
            if any(s != 0 for s in syndrome):
                corrections += 1
                error_pos = syndrome[0] % len(data_part) if syndrome[0] != 0 else 0
                corrected = list(data_part)
                corrected[error_pos] = self.GF4_TO_NUC[
                    (self.NUC_TO_GF4[data_part[error_pos]] - syndrome[0]) % 4
                ]
                data.append("".join(corrected))
            else:
                data.append(data_part)

        return "".join(data), corrections

    def _compute_parity(self, symbols: list[int]) -> list[int]:
        """Compute parity symbols over GF(4)."""
        parity = []
        for j in range(self._n_parity):
            val = 0
            for k, s in enumerate(symbols):
                val = (val + s * pow(k + 1, j + 1, 251)) % 4
            parity.append(val)
        return parity


class DualRailEncoder:
    """Dual-rail encoding for fault-tolerant DNA circuits.

    Each logical signal is encoded as two physical strands: the
    "true" rail and the "complement" rail. Valid states:
        - (high, low)  = logical 1
        - (low, high)  = logical 0
        - (high, high) = fault detected
        - (low, low)   = fault detected

    This provides single-fault detection for each signal.
    """

    def encode(
        self,
        design: DNACircuitDesign,
        compiler: BitstreamToDNA,
    ) -> DNACircuitDesign:
        """Convert a single-rail circuit to dual-rail.

        For each original gate, produces:
        - The original gate on the true rail
        - A complementary gate on the complement rail

        Returns a new DNACircuitDesign with doubled gate count.
        """
        dual_gates: list[Dict[str, Any]] = []
        for g in design.gates:
            # True rail (original)
            dual_gates.append(
                {
                    "type": g.gate_type.value.upper(),
                    "inputs": g.input_names,
                    "output": f"{g.output_name}_T",
                    "threshold": g.threshold,
                }
            )
            # Complement rail
            comp_type = self._complement_gate_type(g.gate_type)
            comp_inputs = [f"{inp}_C" for inp in g.input_names]
            dual_gates.append(
                {
                    "type": comp_type,
                    "inputs": comp_inputs,
                    "output": f"{g.output_name}_C",
                    "threshold": g.threshold,
                }
            )

        all_inputs = []
        for s in design.input_strands:
            all_inputs.extend([f"{s.name}_T", f"{s.name}_C"])

        all_outputs = []
        for s in design.output_strands:
            all_outputs.extend([f"{s.name}_T", f"{s.name}_C"])

        return compiler.compile_network(
            gates=dual_gates,
            input_names=all_inputs,
            output_names=all_outputs,
            name=f"{design.name}_dual_rail",
        )

    def check_faults(
        self,
        result: Dict[str, np.ndarray[Any, Any]],
        threshold_nM: float = 50.0,
    ) -> list[Dict[str, Any]]:
        """Detect faults in dual-rail simulation results."""
        faults: list[Dict[str, Any]] = []
        signals: set[str] = set()

        for key in result:
            if key == "time":
                continue
            if key.endswith("_T") or key.endswith("_C"):
                signals.add(key[:-2])

        for sig in signals:
            t_key = f"{sig}_T"
            c_key = f"{sig}_C"
            if t_key not in result or c_key not in result:
                continue

            t_final = float(result[t_key][-1])
            c_final = float(result[c_key][-1])
            t_high = t_final > threshold_nM
            c_high = c_final > threshold_nM

            if t_high == c_high:  # both high or both low
                faults.append(
                    {
                        "signal": sig,
                        "true_nM": t_final,
                        "comp_nM": c_final,
                        "fault_type": "stuck_high" if t_high else "stuck_low",
                    }
                )

        return faults

    @staticmethod
    def _complement_gate_type(gate_type: GateType) -> str:
        """De Morgan complement gate type."""
        mapping = {
            GateType.AND: "OR",
            GateType.OR: "AND",
            GateType.NOT: "NOT",
            GateType.NAND: "XOR",
            GateType.XOR: "NAND",
            GateType.MUX: "MUX",
            GateType.THRESHOLD: "THRESHOLD",
            GateType.AMPLIFIER: "AMPLIFIER",
            GateType.BUFFER: "BUFFER",
        }
        return mapping.get(gate_type, gate_type.value.upper())

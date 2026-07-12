# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — DNA mapper contracts and physical constants

"""Stable molecular-circuit contracts and thermodynamic constants."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict


_TOEHOLD_LENGTH = 6
_RECOGNITION_LENGTH = 15
_CLAMP_LENGTH = 5
_GC_TARGET_LOW = 0.40
_GC_TARGET_HIGH = 0.60
_MAX_HOMOPOLYMER = 3
_DEFAULT_TEMPERATURE_C = 37.0
_R_GAS = 1.987e-3  # kcal/(mol·K)

# Nearest-neighbour ΔG° (kcal/mol) for DNA at 1 M NaCl, 37 °C
# SantaLucia (1998), Table 2
_NN_DG: Dict[str, float] = {
    "AA": -1.00,
    "TT": -1.00,
    "AT": -0.88,
    "TA": -0.58,
    "CA": -1.45,
    "TG": -1.45,
    "GT": -1.44,
    "AC": -1.44,
    "CT": -1.28,
    "AG": -1.28,
    "GA": -1.30,
    "TC": -1.30,
    "CG": -2.17,
    "GC": -2.24,
    "GG": -1.84,
    "CC": -1.84,
}
_NN_INIT_DG = 1.96  # initiation penalty kcal/mol
_NN_DH: Dict[str, float] = {
    "AA": -7.9,
    "TT": -7.9,
    "AT": -7.2,
    "TA": -7.2,
    "CA": -8.5,
    "TG": -8.5,
    "GT": -8.4,
    "AC": -8.4,
    "CT": -7.8,
    "AG": -7.8,
    "GA": -8.2,
    "TC": -8.2,
    "CG": -10.6,
    "GC": -9.8,
    "GG": -8.0,
    "CC": -8.0,
}
_NN_DS: Dict[str, float] = {
    "AA": -22.2,
    "TT": -22.2,
    "AT": -20.4,
    "TA": -21.3,
    "CA": -22.7,
    "TG": -22.7,
    "GT": -22.4,
    "AC": -22.4,
    "CT": -21.0,
    "AG": -21.0,
    "GA": -22.2,
    "TC": -22.2,
    "CG": -27.2,
    "GC": -24.4,
    "GG": -19.9,
    "CC": -19.9,
}
_NN_INIT_DH = 0.2  # kcal/mol, SantaLucia helix initiation
_NN_INIT_DS = -5.7  # cal/(mol·K), SantaLucia helix initiation
_MIN_HAIRPIN_LOOP_NT = 3
_WC_PAIR_DG: Dict[str, float] = {
    "AT": -1.0,
    "TA": -1.0,
    "GC": -2.0,
    "CG": -2.0,
}
_STACKING_BONUS_DG = -0.35
_HAIRPIN_LOOP_INIT_DG = 1.2
_HAIRPIN_LOOP_SLOPE_DG = 0.15


class GateType(Enum):
    """Supported DNA logic gate types."""

    AND = "and"
    OR = "or"
    NOT = "not"
    NAND = "nand"
    XOR = "xor"
    MUX = "mux"
    THRESHOLD = "threshold"
    CATALYTIC = "catalytic"
    AMPLIFIER = "amplifier"
    BUFFER = "buffer"


class CompilationMethod(Enum):
    """Compilation target for the DNA circuit."""

    DISPLACEMENT = "displacement"
    ENZYMATIC = "enzymatic"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class DNAStrand:
    """A single-stranded DNA molecule used in a circuit.

    Attributes
    ----------
    name : str
        Unique identifier (e.g. ``"gate_0_input_a"``).
    sequence : str
        5' → 3' nucleotide sequence (A, C, G, T).
    role : str
        Functional role: ``"signal"``, ``"fuel"``, ``"output"``,
        ``"waste"``, ``"toehold"``, ``"translator"``.
    concentration_nM : float
        Initial concentration in nanomolar.
    """

    name: str
    sequence: str
    role: str = "signal"
    concentration_nM: float = 100.0

    @property
    def length(self) -> int:
        """Return the nucleotide length of this strand."""
        return len(self.sequence)

    @property
    def gc_content(self) -> float:
        """Return the fraction of nucleotides that are G or C bases."""
        if not self.sequence:
            return 0.0
        gc = sum(1 for c in self.sequence if c in "GC")
        return gc / len(self.sequence)

    @property
    def complement(self) -> str:
        """Return the reverse-complement strand sequence."""
        table = str.maketrans("ACGT", "TGCA")
        return self.sequence.translate(table)[::-1]

    @property
    def max_homopolymer_run(self) -> int:
        """Return the longest repeated-base run in the strand."""
        if not self.sequence:
            return 0
        max_run = 1
        current_run = 1
        for i in range(1, len(self.sequence)):
            if self.sequence[i] == self.sequence[i - 1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1
        return max_run

    def delta_g_37(self) -> float:
        """Nearest-neighbour ΔG° at 37 °C (kcal/mol)."""
        if len(self.sequence) < 2:
            return 0.0
        dg = _NN_INIT_DG
        for i in range(len(self.sequence) - 1):
            dinuc = self.sequence[i : i + 2]
            dg += _NN_DG.get(dinuc, -1.0)
        return dg

    def melting_temperature(self, na_conc_M: float = 1.0, strand_conc_M: float = 2.5e-7) -> float:
        """Return nearest-neighbour DNA duplex melting temperature in °C."""
        if not math.isfinite(float(na_conc_M)) or float(na_conc_M) <= 0.0:
            raise ValueError("na_conc_M must be finite and positive")
        if not math.isfinite(float(strand_conc_M)) or float(strand_conc_M) <= 0.0:
            raise ValueError("strand_conc_M must be finite and positive")
        n = len(self.sequence)
        if n < 2:
            raise ValueError("melting_temperature requires at least two nucleotides")

        delta_h = _NN_INIT_DH
        delta_s = _NN_INIT_DS
        if self.sequence[0] in "AT":
            delta_h += 2.2
            delta_s += 6.9
        else:
            delta_h += 0.1
            delta_s -= 2.8
        if self.sequence[-1] in "AT":
            delta_h += 2.2
            delta_s += 6.9
        else:
            delta_h += 0.1
            delta_s -= 2.8

        for i in range(n - 1):
            dinuc = self.sequence[i : i + 2]
            delta_h += _NN_DH[dinuc]
            delta_s += _NN_DS[dinuc]

        tm_kelvin = (1000.0 * delta_h) / (
            delta_s + (1000.0 * _R_GAS) * math.log(float(strand_conc_M) / 4.0)
        )
        salt_correction_c = 16.6 * math.log10(float(na_conc_M))
        return tm_kelvin - 273.15 + salt_correction_c


@dataclass
class DNAGate:
    """A logic gate implemented via DNA strand displacement.

    Attributes
    ----------
    gate_id : int
        Unique gate index in the circuit.
    gate_type : GateType
        Logic operation (AND, OR, NOT, etc.).
    input_names : list[str]
        Names of input signal strands.
    output_name : str
        Name of the output signal strand.
    strands : list[DNAStrand]
        All DNA strands participating in this gate (inputs, fuel,
        translator complexes, output, waste).
    threshold : float
        For threshold gates, the activation threshold concentration.
    leak_rate : float
        Estimated spurious activation rate (per second).
    """

    gate_id: int
    gate_type: GateType
    input_names: list[str]
    output_name: str
    strands: list[DNAStrand] = field(default_factory=list)
    threshold: float = 0.0
    leak_rate: float = 1e-9

    @property
    def strand_count(self) -> int:
        """Return the number of strands implementing this gate."""
        return len(self.strands)


@dataclass
class DNACircuitDesign:
    """Complete compiled DNA circuit.

    Holds the full strand-level design for a compiled SC network,
    including all gates, signal routing, and thermodynamic validation.

    Attributes
    ----------
    name : str
        Circuit identifier.
    gates : list[DNAGate]
        Ordered list of compiled gates.
    input_strands : list[DNAStrand]
        Primary input signal strands.
    output_strands : list[DNAStrand]
        Primary output signal strands.
    fuel_strands : list[DNAStrand]
        Fuel/helper strands consumed during computation.
    method : CompilationMethod
        Compilation target used.
    temperature_c : float
        Design temperature in Celsius.
    na_concentration_M : float
        Sodium concentration for thermodynamic calculations.
    """

    name: str = "sc_dna_circuit"
    gates: list[DNAGate] = field(default_factory=list)
    input_strands: list[DNAStrand] = field(default_factory=list)
    output_strands: list[DNAStrand] = field(default_factory=list)
    fuel_strands: list[DNAStrand] = field(default_factory=list)
    method: CompilationMethod = CompilationMethod.DISPLACEMENT
    temperature_c: float = _DEFAULT_TEMPERATURE_C
    na_concentration_M: float = 1.0

    @property
    def total_strands(self) -> int:
        """Return the total strand count across circuit inputs, outputs, fuel, and gates."""
        return (
            len(self.input_strands)
            + len(self.output_strands)
            + len(self.fuel_strands)
            + sum(g.strand_count for g in self.gates)
        )

    @property
    def total_gates(self) -> int:
        """Return the number of DNA logic gates in the circuit."""
        return len(self.gates)

    @property
    def total_nucleotides(self) -> int:
        """Return the total nucleotide count across all circuit strands."""
        count = 0
        for s in self.input_strands + self.output_strands + self.fuel_strands:
            count += s.length
        for g in self.gates:
            for s in g.strands:
                count += s.length
        return count

    def validate(self) -> list[str]:
        """Run design rule checks. Returns list of warnings."""
        warnings: list[str] = []
        all_strands = self.input_strands + self.output_strands + self.fuel_strands
        for g in self.gates:
            all_strands.extend(g.strands)

        for s in all_strands:
            if not (_GC_TARGET_LOW <= s.gc_content <= _GC_TARGET_HIGH):
                warnings.append(
                    f"{s.name}: GC content {s.gc_content:.2f} outside "
                    f"[{_GC_TARGET_LOW}, {_GC_TARGET_HIGH}]"
                )
            if s.max_homopolymer_run > _MAX_HOMOPOLYMER:
                warnings.append(
                    f"{s.name}: homopolymer run {s.max_homopolymer_run} "
                    f"exceeds max {_MAX_HOMOPOLYMER}"
                )
        return warnings

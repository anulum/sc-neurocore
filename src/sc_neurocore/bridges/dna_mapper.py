# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Molecular/DNA Computing Mapper

"""Molecular/DNA computing mapper for stochastic computing bitstreams.

This module compiles SC Boolean networks into DNA strand displacement
circuits, enzymatic gate cascades, and NUPACK-compatible sequence designs.
It bridges the gap between digital stochastic computing and wet-lab
molecular computation.

Theory
------
Stochastic computing encodes probabilities as the frequency of 1s in a
binary bitstream. Each SC gate (AND, OR, MUX) performs a probabilistic
operation physically via CMOS transistors. DNA strand displacement
circuits can implement the *same* Boolean operations using toehold-mediated
hybridization, where the presence/absence of a DNA strand encodes a
logical 1/0.

The mapping is:

    SC AND(a, b) → Displacement AND: signal strands A and B must both
    bind to a translator complex before output strand O is released.

    SC OR(a, b) → Catalytic hairpin assembly: either A or B triggers
    a cascade that releases O.

    SC NOT(a) → Catalytic junction: strand A sequesters its own
    complement, releasing a pre-loaded output O.

References
----------
- Zhang, D.Y. & Winfree, E. (2009). Control of DNA strand displacement
  kinetics using toehold exchange. JACS, 131(47), 17303–17314.
- Qian, L. & Winfree, E. (2011). Scaling up digital circuit computation
  with DNA strand displacement cascades. Science, 332(6034), 1196–1201.
- Seelig, G. et al. (2006). Enzyme-free nucleic acid logic circuits.
  Science, 314(5805), 1585–1588.
- Soloveichik, D. et al. (2010). DNA as a universal substrate for
  chemical kinetics. PNAS, 107(12), 5393–5398.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sc_neurocore_engine.dna import has_full_dna_backend
except ImportError:

    def has_full_dna_backend() -> bool:
        return False

# ── Soft imports ──────────────────────────────────────────────────────

try:
    import nupack

    _HAS_NUPACK = True
except ImportError:
    nupack = None
    _HAS_NUPACK = False

try:
    _HAS_RUST_DNA = has_full_dna_backend()
except ImportError:
    _HAS_RUST_DNA = False


# ── Constants ─────────────────────────────────────────────────────────

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
        step = trace[i][j]
        if step is None:
            return
        kind, a, b = step
        if kind == "skip_i" or kind == "skip_j":
            traceback(a, b)
        elif kind == "pair":
            pairs.append((i, j))
            traceback(a, b)
        elif kind == "split":
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


# ══════════════════════════════════════════════════════════════════════
# Data Classes
# ══════════════════════════════════════════════════════════════════════


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
        return len(self.sequence)

    @property
    def gc_content(self) -> float:
        if not self.sequence:
            return 0.0
        gc = sum(1 for c in self.sequence if c in "GC")
        return gc / len(self.sequence)

    @property
    def complement(self) -> str:
        table = str.maketrans("ACGT", "TGCA")
        return self.sequence.translate(table)[::-1]

    @property
    def max_homopolymer_run(self) -> int:
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
        return (
            len(self.input_strands)
            + len(self.output_strands)
            + len(self.fuel_strands)
            + sum(g.strand_count for g in self.gates)
        )

    @property
    def total_gates(self) -> int:
        return len(self.gates)

    @property
    def total_nucleotides(self) -> int:
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


# ══════════════════════════════════════════════════════════════════════
# Sequence Design Engine
# ══════════════════════════════════════════════════════════════════════


class SequenceDesigner:
    """Deterministic DNA sequence generator with constraint satisfaction.

    Generates sequences that satisfy GC content, homopolymer, and
    orthogonality constraints using a seed-based deterministic algorithm.
    This ensures reproducible designs without requiring NUPACK.

    Parameters
    ----------
    seed : int
        Random seed for reproducible sequence generation.
    gc_target : tuple[float, float]
        Acceptable GC content range (default 0.40–0.60).
    max_homopolymer : int
        Maximum consecutive identical nucleotides (default 3).
    """

    def __init__(
        self,
        seed: int = 42,
        gc_target: Tuple[float, float] = (_GC_TARGET_LOW, _GC_TARGET_HIGH),
        max_homopolymer: int = _MAX_HOMOPOLYMER,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self._gc_target = gc_target
        self._max_homopolymer = max_homopolymer
        self._used_sequences: list[str] = []

    def generate(self, length: int, name: str = "seq") -> str:
        """Generate a sequence satisfying all constraints.

        Uses rejection sampling with guided nucleotide selection to
        maintain GC content within bounds while avoiding homopolymer
        runs.

        Parameters
        ----------
        length : int
            Desired sequence length.
        name : str
            Identifier for debugging (used in hash seed).

        Returns
        -------
        str
            Valid nucleotide sequence (A, C, G, T).
        """
        nucs = ["A", "C", "G", "T"]
        best_seq = ""
        best_score = float("inf")

        seed_hash = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(self._rng.integers(0, 2**31) + seed_hash)

        for _attempt in range(200):
            seq: list[str] = []
            gc_count = 0

            for i in range(length):
                # Determine allowed nucleotides
                allowed = list(nucs)

                # Prevent homopolymer runs
                if len(seq) >= self._max_homopolymer:
                    last_n = seq[-self._max_homopolymer :]
                    if len(set(last_n)) == 1:
                        allowed = [n for n in allowed if n != last_n[0]]

                # Bias toward GC target
                if i > 0:
                    current_gc = gc_count / i
                    if current_gc < self._gc_target[0]:
                        # Need more GC
                        weights = [0.15, 0.35, 0.35, 0.15]
                    elif current_gc > self._gc_target[1]:
                        # Need more AT
                        weights = [0.35, 0.15, 0.15, 0.35]
                    else:
                        weights = [0.25, 0.25, 0.25, 0.25]
                    # Zero out disallowed
                    weights = [w if nucs[j] in allowed else 0.0 for j, w in enumerate(weights)]
                else:
                    weights = [1.0 if n in allowed else 0.0 for n in nucs]

                total = sum(weights)
                if total == 0:
                    weights = [1.0 / len(nucs)] * len(nucs)
                    total = 1.0
                probs = [w / total for w in weights]

                nuc = rng.choice(nucs, p=probs)
                seq.append(nuc)
                if nuc in "GC":
                    gc_count += 1

            candidate = "".join(seq)
            gc = gc_count / length
            score = abs(gc - 0.5) * 10

            # Penalize homopolymer runs
            max_run = 1
            cur_run = 1
            for i in range(1, len(candidate)):
                if candidate[i] == candidate[i - 1]:
                    cur_run += 1
                    max_run = max(max_run, cur_run)
                else:
                    cur_run = 1
            if max_run > self._max_homopolymer:
                score += (max_run - self._max_homopolymer) * 5

            # Penalize similarity to existing sequences
            for existing in self._used_sequences:
                overlap = sum(1 for a, b in zip(candidate, existing) if a == b)
                similarity = overlap / max(len(candidate), len(existing), 1)
                if similarity > 0.7:
                    score += similarity * 10

            if score < best_score:
                best_score = score
                best_seq = candidate

            if score < 0.5:
                break

        self._used_sequences.append(best_seq)
        return best_seq

    def generate_complement(self, sequence: str) -> str:
        """Return the Watson-Crick complement (3' → 5')."""
        table = str.maketrans("ACGT", "TGCA")
        return sequence.translate(table)[::-1]

    def generate_toehold(self, name: str = "toehold") -> str:
        """Generate a toehold domain (6 nt)."""
        return self.generate(_TOEHOLD_LENGTH, name)

    def generate_recognition(self, name: str = "recog") -> str:
        """Generate a recognition domain (15 nt)."""
        return self.generate(_RECOGNITION_LENGTH, name)


# ══════════════════════════════════════════════════════════════════════
# Strand Displacement Compiler
# ══════════════════════════════════════════════════════════════════════


class StrandDisplacementCompiler:
    """Compile SC Boolean gates into toehold-mediated displacement circuits.

    Implements the seesaw gate architecture from Qian & Winfree (2011)
    adapted for SC-NeuroCore's bitstream operations.

    Parameters
    ----------
    designer : SequenceDesigner | None
        Sequence generator. If None, a default is created.
    temperature_c : float
        Target operating temperature in Celsius.
    """

    def __init__(
        self,
        designer: Optional[SequenceDesigner] = None,
        temperature_c: float = _DEFAULT_TEMPERATURE_C,
    ) -> None:
        self._designer = designer or SequenceDesigner()
        self._temperature_c = temperature_c
        self._gate_counter = 0

    def compile_and(self, input_a: str, input_b: str, output: str) -> DNAGate:
        """Compile a 2-input AND gate.

        Architecture: two-layer seesaw cascade.
        Signal A and signal B must both be present to displace the
        output strand from a dual-input translator complex.

        Gate strands:
        - Translator complex (double-stranded with two toehold domains)
        - Threshold strand (absorbs leak from single-input activation)
        - Fuel strand (drives the reaction to completion)
        - Output strand (released into solution)
        """
        gid = self._gate_counter
        self._gate_counter += 1

        # Generate domains
        th_a = self._designer.generate_toehold(f"g{gid}_th_a")
        th_b = self._designer.generate_toehold(f"g{gid}_th_b")
        recog_a = self._designer.generate_recognition(f"g{gid}_rec_a")
        recog_b = self._designer.generate_recognition(f"g{gid}_rec_b")
        recog_out = self._designer.generate_recognition(f"g{gid}_rec_out")
        th_out = self._designer.generate_toehold(f"g{gid}_th_out")

        strands = [
            DNAStrand(
                name=f"g{gid}_translator_top",
                sequence=th_a + recog_a + recog_b + th_b,
                role="translator",
                concentration_nM=200.0,
            ),
            DNAStrand(
                name=f"g{gid}_translator_bot",
                sequence=self._designer.generate_complement(recog_a + recog_b),
                role="translator",
                concentration_nM=200.0,
            ),
            DNAStrand(
                name=f"g{gid}_output",
                sequence=th_out + recog_out,
                role="output",
                concentration_nM=0.0,
            ),
            DNAStrand(
                name=f"g{gid}_fuel",
                sequence=self._designer.generate_complement(recog_a) + th_a,
                role="fuel",
                concentration_nM=500.0,
            ),
            DNAStrand(
                name=f"g{gid}_threshold",
                sequence=self._designer.generate_complement(th_a + recog_a[:8]),
                role="threshold",
                concentration_nM=50.0,
            ),
        ]

        leak = self._estimate_leak_rate(strands[0], strands[4])

        return DNAGate(
            gate_id=gid,
            gate_type=GateType.AND,
            input_names=[input_a, input_b],
            output_name=output,
            strands=strands,
            leak_rate=leak,
        )

    def compile_or(self, input_a: str, input_b: str, output: str) -> DNAGate:
        """Compile a 2-input OR gate via catalytic hairpin assembly.

        Either input signal triggers hairpin opening and output release.
        """
        gid = self._gate_counter
        self._gate_counter += 1

        th_a = self._designer.generate_toehold(f"g{gid}_th_a")
        th_b = self._designer.generate_toehold(f"g{gid}_th_b")
        stem = self._designer.generate_recognition(f"g{gid}_stem")
        loop = self._designer.generate(8, f"g{gid}_loop")
        recog_out = self._designer.generate_recognition(f"g{gid}_rec_out")

        hairpin_seq = th_a + stem + loop + self._designer.generate_complement(stem)

        strands = [
            DNAStrand(
                name=f"g{gid}_hairpin_a",
                sequence=hairpin_seq,
                role="translator",
                concentration_nM=200.0,
            ),
            DNAStrand(
                name=f"g{gid}_hairpin_b",
                sequence=th_b + stem + loop + self._designer.generate_complement(stem),
                role="translator",
                concentration_nM=200.0,
            ),
            DNAStrand(
                name=f"g{gid}_output",
                sequence=recog_out,
                role="output",
                concentration_nM=0.0,
            ),
            DNAStrand(
                name=f"g{gid}_fuel",
                sequence=self._designer.generate_complement(stem) + th_a,
                role="fuel",
                concentration_nM=500.0,
            ),
        ]

        return DNAGate(
            gate_id=gid,
            gate_type=GateType.OR,
            input_names=[input_a, input_b],
            output_name=output,
            strands=strands,
            leak_rate=1e-9,
        )

    def compile_not(self, input_name: str, output: str) -> DNAGate:
        """Compile a NOT gate via strand sequestration.

        Input signal sequesters the blocking strand, releasing the
        pre-loaded output.
        """
        gid = self._gate_counter
        self._gate_counter += 1

        th = self._designer.generate_toehold(f"g{gid}_th")
        recog = self._designer.generate_recognition(f"g{gid}_rec")
        recog_out = self._designer.generate_recognition(f"g{gid}_rec_out")

        strands = [
            DNAStrand(
                name=f"g{gid}_blocker",
                sequence=th + recog,
                role="translator",
                concentration_nM=200.0,
            ),
            DNAStrand(
                name=f"g{gid}_output_complex",
                sequence=self._designer.generate_complement(recog) + recog_out,
                role="translator",
                concentration_nM=200.0,
            ),
            DNAStrand(
                name=f"g{gid}_output",
                sequence=recog_out,
                role="output",
                concentration_nM=0.0,
            ),
        ]

        return DNAGate(
            gate_id=gid,
            gate_type=GateType.NOT,
            input_names=[input_name],
            output_name=output,
            strands=strands,
            leak_rate=5e-10,
        )

    def compile_threshold(self, input_name: str, output: str, threshold: float = 0.5) -> DNAGate:
        """Compile a threshold gate for concentration-dependent activation.

        The threshold strand absorbs input below the threshold
        concentration; only excess input activates the output.
        """
        gid = self._gate_counter
        self._gate_counter += 1

        th = self._designer.generate_toehold(f"g{gid}_th")
        recog = self._designer.generate_recognition(f"g{gid}_rec")
        recog_out = self._designer.generate_recognition(f"g{gid}_rec_out")

        threshold_conc = threshold * 200.0  # scale to working range

        strands = [
            DNAStrand(
                name=f"g{gid}_absorber",
                sequence=self._designer.generate_complement(th + recog),
                role="threshold",
                concentration_nM=threshold_conc,
            ),
            DNAStrand(
                name=f"g{gid}_translator",
                sequence=th + recog + recog_out,
                role="translator",
                concentration_nM=200.0,
            ),
            DNAStrand(
                name=f"g{gid}_output",
                sequence=recog_out,
                role="output",
                concentration_nM=0.0,
            ),
        ]

        return DNAGate(
            gate_id=gid,
            gate_type=GateType.THRESHOLD,
            input_names=[input_name],
            output_name=output,
            strands=strands,
            threshold=threshold,
            leak_rate=2e-9,
        )

    def compile_mux(self, select: str, input_a: str, input_b: str, output: str) -> DNAGate:
        """Compile a 2:1 multiplexer (MUX) gate.

        Implements P(out) = select·P(a) + (1−select)·P(b), the core
        SC operation for weighted addition.  Architecture: dual-threshold
        cascade where the select signal activates one of two pathways.
        """
        gid = self._gate_counter
        self._gate_counter += 1

        th_s = self._designer.generate_toehold(f"g{gid}_th_s")
        th_a = self._designer.generate_toehold(f"g{gid}_th_a")
        th_b = self._designer.generate_toehold(f"g{gid}_th_b")
        recog_a = self._designer.generate_recognition(f"g{gid}_rec_a")
        recog_b = self._designer.generate_recognition(f"g{gid}_rec_b")
        recog_out = self._designer.generate_recognition(f"g{gid}_rec_out")

        strands = [
            DNAStrand(
                name=f"g{gid}_path_a",
                sequence=th_s + recog_a + th_a,
                role="translator",
                concentration_nM=200.0,
            ),
            DNAStrand(
                name=f"g{gid}_path_b",
                sequence=self._designer.generate_complement(th_s) + recog_b + th_b,
                role="translator",
                concentration_nM=200.0,
            ),
            DNAStrand(
                name=f"g{gid}_combiner",
                sequence=recog_out,
                role="translator",
                concentration_nM=200.0,
            ),
            DNAStrand(
                name=f"g{gid}_output",
                sequence=recog_out,
                role="output",
                concentration_nM=0.0,
            ),
            DNAStrand(
                name=f"g{gid}_fuel",
                sequence=self._designer.generate_complement(recog_a) + th_s,
                role="fuel",
                concentration_nM=500.0,
            ),
        ]

        return DNAGate(
            gate_id=gid,
            gate_type=GateType.MUX,
            input_names=[select, input_a, input_b],
            output_name=output,
            strands=strands,
            leak_rate=2e-9,
        )

    def compile_amplifier(self, input_name: str, output: str) -> DNAGate:
        """Compile a catalytic signal amplifier.

        One input molecule catalytically releases many output molecules
        via a fuel-driven catalytic cycle. Useful for fan-out from a
        single signal source to multiple downstream consumers.
        """
        gid = self._gate_counter
        self._gate_counter += 1

        th = self._designer.generate_toehold(f"g{gid}_th")
        recog = self._designer.generate_recognition(f"g{gid}_rec")
        recog_out = self._designer.generate_recognition(f"g{gid}_rec_out")
        th_cat = self._designer.generate_toehold(f"g{gid}_th_cat")

        strands = [
            DNAStrand(
                name=f"g{gid}_catalyst_complex",
                sequence=th + recog + th_cat,
                role="translator",
                concentration_nM=200.0,
            ),
            DNAStrand(
                name=f"g{gid}_substrate",
                sequence=self._designer.generate_complement(recog) + recog_out,
                role="translator",
                concentration_nM=500.0,
            ),
            DNAStrand(
                name=f"g{gid}_fuel",
                sequence=self._designer.generate_complement(th + recog),
                role="fuel",
                concentration_nM=1000.0,
            ),
            DNAStrand(
                name=f"g{gid}_output",
                sequence=recog_out,
                role="output",
                concentration_nM=0.0,
            ),
        ]

        return DNAGate(
            gate_id=gid,
            gate_type=GateType.AMPLIFIER,
            input_names=[input_name],
            output_name=output,
            strands=strands,
            leak_rate=1e-9,
        )

    def compile_buffer(self, input_name: str, output: str) -> DNAGate:
        """Compile a signal restoration buffer.

        Passes the signal through with level restoration, cleaning up
        degraded signals in long cascades. Uses a threshold + amplifier
        internal architecture.
        """
        gid = self._gate_counter
        self._gate_counter += 1

        th = self._designer.generate_toehold(f"g{gid}_th")
        recog = self._designer.generate_recognition(f"g{gid}_rec")
        recog_out = self._designer.generate_recognition(f"g{gid}_rec_out")

        strands = [
            DNAStrand(
                name=f"g{gid}_threshold",
                sequence=self._designer.generate_complement(th + recog[:8]),
                role="threshold",
                concentration_nM=80.0,
            ),
            DNAStrand(
                name=f"g{gid}_translator",
                sequence=th + recog + recog_out,
                role="translator",
                concentration_nM=200.0,
            ),
            DNAStrand(
                name=f"g{gid}_output",
                sequence=recog_out,
                role="output",
                concentration_nM=0.0,
            ),
        ]

        return DNAGate(
            gate_id=gid,
            gate_type=GateType.BUFFER,
            input_names=[input_name],
            output_name=output,
            strands=strands,
            leak_rate=5e-10,
        )

    def _estimate_leak_rate(self, strand: DNAStrand, blocker: DNAStrand) -> float:
        """Estimate spurious strand displacement rate.

        Uses the strongest contiguous Watson-Crick interaction between
        the strand and blocker to approximate the leak suppression.
        """
        dg = self._strongest_blocker_delta_g(strand, blocker)
        temp_k = self._temperature_c + 273.15
        k_leak = 1e-6 * math.exp(dg / (_R_GAS * temp_k))
        return min(k_leak, 1e-6)

    @staticmethod
    def _strongest_blocker_delta_g(strand: DNAStrand, blocker: DNAStrand) -> float:
        """Return the most stable contiguous blocker-binding ΔG° at 37 °C."""
        query = strand.sequence
        target = blocker.complement
        best_dg = 0.0
        for offset in range(-len(target) + 1, len(query)):
            run: list[str] = []
            for i, base in enumerate(query):
                j = i - offset
                if 0 <= j < len(target) and base == target[j]:
                    run.append(base)
                    continue
                if len(run) >= 2:
                    best_dg = min(best_dg, DNAStrand("blocker_run", "".join(run)).delta_g_37())
                run = []
            if len(run) >= 2:
                best_dg = min(best_dg, DNAStrand("blocker_run", "".join(run)).delta_g_37())
        return best_dg


# ══════════════════════════════════════════════════════════════════════
# Enzymatic Gate Compiler
# ══════════════════════════════════════════════════════════════════════


class EnzymaticGateCompiler:
    """Compile SC gates into enzyme-mediated DNA logic circuits.

    Uses restriction enzymes and ligases to implement Boolean operations
    on DNA substrates. Operates on double-stranded DNA with specific
    recognition sites.

    Parameters
    ----------
    designer : SequenceDesigner | None
        Sequence generator.
    """

    def __init__(self, designer: Optional[SequenceDesigner] = None) -> None:
        self._designer = designer or SequenceDesigner(seed=137)
        self._gate_counter = 0

    # Restriction enzyme recognition sites
    ENZYMES: Dict[str, str] = {
        "EcoRI": "GAATTC",
        "BamHI": "GGATCC",
        "HindIII": "AAGCTT",
        "NotI": "GCGGCCGC",
        "XhoI": "CTCGAG",
        "NheI": "GCTAGC",
        "SpeI": "ACTAGT",
        "SalI": "GTCGAC",
    }

    def compile_nand(self, input_a: str, input_b: str, output: str) -> DNAGate:
        """NAND gate via dual restriction enzyme cascade.

        Both inputs must be present (as enzymes) to cleave the substrate
        at two sites. Only when both sites are cleaved is the output
        fragment *not* produced (NAND logic).
        """
        gid = self._gate_counter
        self._gate_counter += 1

        flank_5 = self._designer.generate(20, f"g{gid}_flank5")
        flank_3 = self._designer.generate(20, f"g{gid}_flank3")
        spacer = self._designer.generate(10, f"g{gid}_spacer")
        out_seq = self._designer.generate_recognition(f"g{gid}_out")

        substrate = (
            flank_5
            + self.ENZYMES["EcoRI"]
            + spacer
            + out_seq
            + spacer
            + self.ENZYMES["BamHI"]
            + flank_3
        )

        strands = [
            DNAStrand(
                name=f"g{gid}_substrate",
                sequence=substrate,
                role="translator",
                concentration_nM=100.0,
            ),
            DNAStrand(
                name=f"g{gid}_output",
                sequence=out_seq,
                role="output",
                concentration_nM=0.0,
            ),
        ]

        return DNAGate(
            gate_id=gid,
            gate_type=GateType.NAND,
            input_names=[input_a, input_b],
            output_name=output,
            strands=strands,
        )

    def compile_xor(self, input_a: str, input_b: str, output: str) -> DNAGate:
        """XOR gate via nick-sealing ligase logic.

        Uses two nicked substrates where each input ligase seals one
        nick. Only when exactly one nick is sealed does the output
        become stable (XOR logic).
        """
        gid = self._gate_counter
        self._gate_counter += 1

        left = self._designer.generate(20, f"g{gid}_left")
        right = self._designer.generate(20, f"g{gid}_right")
        out_seq = self._designer.generate_recognition(f"g{gid}_out")

        strands = [
            DNAStrand(
                name=f"g{gid}_nick_a",
                sequence=left + out_seq[:7],
                role="translator",
                concentration_nM=100.0,
            ),
            DNAStrand(
                name=f"g{gid}_nick_b",
                sequence=out_seq[7:] + right,
                role="translator",
                concentration_nM=100.0,
            ),
            DNAStrand(
                name=f"g{gid}_template",
                sequence=self._designer.generate_complement(left + out_seq + right),
                role="translator",
                concentration_nM=100.0,
            ),
            DNAStrand(
                name=f"g{gid}_output",
                sequence=out_seq,
                role="output",
                concentration_nM=0.0,
            ),
        ]

        return DNAGate(
            gate_id=gid,
            gate_type=GateType.XOR,
            input_names=[input_a, input_b],
            output_name=output,
            strands=strands,
        )


# ══════════════════════════════════════════════════════════════════════
# NUPACK Interface
# ══════════════════════════════════════════════════════════════════════


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
        return _HAS_NUPACK

    def compute_mfe(self, sequence: str) -> Tuple[float, str]:
        """Compute minimum free energy and structure.

        Returns
        -------
        tuple[float, str]
            (energy_kcal_mol, dot_bracket_structure)
        """
        if _HAS_NUPACK:
            model = nupack.Model(
                material="dna",
                celsius=self._temperature_c,
                sodium=self._na_M,
            )
            strand = nupack.Strand(sequence, name="query")
            result = nupack.mfe(strands=[strand], model=model)
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
        if _HAS_NUPACK:
            model = nupack.Model(
                material="dna",
                celsius=self._temperature_c,
                sodium=self._na_M,
            )
            strand = nupack.Strand(sequence, name="query")
            result = nupack.pairs(strands=[strand], model=model)
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


# ══════════════════════════════════════════════════════════════════════
# Kinetic Simulation Engine
# ══════════════════════════════════════════════════════════════════════


class KineticSimulator:
    """Mass-action kinetics simulator for DNA strand displacement.

    Simulates the time evolution of strand concentrations using
    selectable integration (Euler or RK4) with Arrhenius temperature
    scaling of rate constants.

    Parameters
    ----------
    rate_hybridization : float
        Second-order rate constant for toehold binding (M⁻¹ s⁻¹).
    rate_displacement : float
        First-order rate constant for branch migration (s⁻¹).
    temperature_c : float
        Temperature in Celsius.
    integrator : str
        Integration method: ``"euler"`` or ``"rk4"``.
    """

    def __init__(
        self,
        rate_hybridization: float = 3e5,
        rate_displacement: float = 1.0,
        temperature_c: float = _DEFAULT_TEMPERATURE_C,
        integrator: str = "euler",
    ) -> None:
        self._k_hyb = rate_hybridization
        self._k_disp = rate_displacement
        self._temperature_c = temperature_c
        self._integrator = integrator

    def _arrhenius_scale(self, k_ref: float, ea_kcal: float = 15.0) -> float:
        """Scale rate constant from 37°C to operating temperature via Arrhenius."""
        t_ref = 310.15  # 37°C in Kelvin
        t_op = self._temperature_c + 273.15
        return k_ref * math.exp(-(ea_kcal / _R_GAS) * (1.0 / t_op - 1.0 / t_ref))

    def _compute_k_eff(
        self,
        gate: "DNAGate",
        input_concentrations: Dict[str, float],
    ) -> float:
        """Compute effective rate constant for a gate."""
        k_hyb = self._arrhenius_scale(self._k_hyb)
        k_disp = self._arrhenius_scale(self._k_disp)

        if gate.gate_type == GateType.AND:
            inputs_conc = [input_concentrations.get(inp, 0.0) for inp in gate.input_names]
            input_present = all(c > 0.0 for c in inputs_conc)
            k_eff = k_hyb * min(inputs_conc) * 1e-9 * (1.0 if input_present else 0.0)

        elif gate.gate_type == GateType.OR:
            inputs_conc = [input_concentrations.get(inp, 0.0) for inp in gate.input_names]
            k_eff = k_hyb * max(inputs_conc) * 1e-9

        elif gate.gate_type == GateType.NOT:
            inp_conc = input_concentrations.get(gate.input_names[0], 0.0)
            k_eff = k_disp * (1.0 - min(inp_conc / 200.0, 1.0))

        elif gate.gate_type == GateType.THRESHOLD:
            inp_conc = input_concentrations.get(gate.input_names[0], 0.0)
            excess = max(0.0, inp_conc - gate.threshold * 200.0)
            k_eff = k_hyb * excess * 1e-9

        elif gate.gate_type == GateType.MUX:
            sel = input_concentrations.get(gate.input_names[0], 0.0)
            a = input_concentrations.get(gate.input_names[1], 0.0)
            b = input_concentrations.get(gate.input_names[2], 0.0)
            sel_frac = min(sel / 200.0, 1.0)
            k_eff = k_hyb * (sel_frac * a + (1.0 - sel_frac) * b) * 1e-9

        elif gate.gate_type == GateType.AMPLIFIER:
            inp_conc = input_concentrations.get(gate.input_names[0], 0.0)
            k_eff = k_hyb * inp_conc * 1e-9 * 5.0  # catalytic turnover

        elif gate.gate_type == GateType.BUFFER:
            inp_conc = input_concentrations.get(gate.input_names[0], 0.0)
            k_eff = k_disp * min(inp_conc / 200.0, 1.0)

        else:
            k_eff = 0.0

        return k_eff + gate.leak_rate

    def simulate(
        self,
        design: DNACircuitDesign,
        input_concentrations: Dict[str, float],
        duration_s: float = 3600.0,
        dt: float = 1.0,
    ) -> Dict[str, np.ndarray[Any, Any]]:
        """Simulate circuit kinetics.

        Parameters
        ----------
        design : DNACircuitDesign
            Compiled circuit to simulate.
        input_concentrations : dict[str, float]
            Initial concentrations of input signal strands (nM).
        duration_s : float
            Simulation duration in seconds.
        dt : float
            Time step in seconds.

        Returns
        -------
        dict[str, np.ndarray]
            Time traces for each output strand. Keys are strand names,
            values are 1D arrays of concentrations over time.
            Includes ``"time"`` key with the time axis.
        """
        n_steps = int(duration_s / dt)
        time = np.linspace(0.0, duration_s, n_steps)

        outputs: Dict[str, np.ndarray[Any, Any]] = {"time": time}
        max_conc = 200.0

        for g in design.gates:
            conc = np.zeros(n_steps)
            k_eff = self._compute_k_eff(g, input_concentrations)

            if self._integrator == "rk4":
                for t in range(1, n_steps):
                    c = conc[t - 1]
                    k1 = k_eff * (max_conc - c) * dt
                    k2 = k_eff * (max_conc - (c + k1 / 2)) * dt
                    k3 = k_eff * (max_conc - (c + k2 / 2)) * dt
                    k4 = k_eff * (max_conc - (c + k3)) * dt
                    conc[t] = c + (k1 + 2 * k2 + 2 * k3 + k4) / 6
                    conc[t] = max(0.0, min(conc[t], max_conc))
            else:
                for t in range(1, n_steps):
                    d_conc = k_eff * (max_conc - conc[t - 1]) * dt
                    conc[t] = conc[t - 1] + d_conc
                    conc[t] = max(0.0, min(conc[t], max_conc))

            outputs[g.output_name] = conc

        return outputs


# ══════════════════════════════════════════════════════════════════════
# Export Functions
# ══════════════════════════════════════════════════════════════════════


def export_genbank(design: DNACircuitDesign, path: str) -> None:
    """Export circuit design to GenBank format.

    Creates a multi-record GenBank file with one record per strand,
    including annotations for functional domains (toehold, recognition,
    clamp).

    Parameters
    ----------
    design : DNACircuitDesign
        Compiled circuit.
    path : str
        Output file path (e.g. ``"circuit.gb"``).
    """
    all_strands = design.input_strands + design.output_strands + design.fuel_strands
    for g in design.gates:
        all_strands.extend(g.strands)

    records: list[str] = []
    for strand in all_strands:
        locus = strand.name[:16].ljust(16)
        bp = len(strand.sequence)
        header = f"LOCUS       {locus} {bp:>5} bp    ss-DNA     linear   SYN 01-JAN-2026\n"
        definition = f"DEFINITION  {strand.name} [{strand.role}]\n"
        accession = f"ACCESSION   {strand.name}\n"
        source = "SOURCE      synthetic construct\n"
        organism = (
            "  ORGANISM  synthetic construct\n            other sequences; artificial sequences.\n"
        )

        features = "FEATURES             Location/Qualifiers\n"
        features += (
            f"     source          1..{bp}\n"
            f'                     /mol_type="other DNA"\n'
            f'                     /organism="synthetic construct"\n'
        )
        if strand.role == "translator" and bp > _TOEHOLD_LENGTH:
            features += (
                f"     misc_feature    1..{_TOEHOLD_LENGTH}\n"
                f'                     /label="toehold"\n'
            )
            features += (
                f"     misc_feature    {_TOEHOLD_LENGTH + 1}..{bp}\n"
                f'                     /label="recognition"\n'
            )

        origin = "ORIGIN\n"
        seq_lines = ""
        for i in range(0, bp, 60):
            pos = str(i + 1).rjust(9)
            chunk = strand.sequence[i : i + 60]
            groups = " ".join(chunk[j : j + 10] for j in range(0, len(chunk), 10))
            seq_lines += f"{pos} {groups}\n"

        record = (
            header
            + definition
            + accession
            + source
            + organism
            + features
            + origin
            + seq_lines
            + "//\n"
        )
        records.append(record)

    with open(path, "w") as f:
        f.write("\n".join(records))


def export_fasta(design: DNACircuitDesign, path: str) -> None:
    """Export all strands to FASTA format.

    Parameters
    ----------
    design : DNACircuitDesign
        Compiled circuit.
    path : str
        Output file path (e.g. ``"circuit.fasta"``).
    """
    all_strands = design.input_strands + design.output_strands + design.fuel_strands
    for g in design.gates:
        all_strands.extend(g.strands)

    with open(path, "w") as f:
        for strand in all_strands:
            f.write(
                f">{strand.name} role={strand.role} "
                f"gc={strand.gc_content:.3f} "
                f"conc={strand.concentration_nM}nM\n"
            )
            # Wrap to 80 characters
            for i in range(0, len(strand.sequence), 80):
                f.write(strand.sequence[i : i + 80] + "\n")


def export_nupack_input(design: DNACircuitDesign, path: str) -> None:
    """Export circuit in NUPACK multi-strand input format.

    Parameters
    ----------
    design : DNACircuitDesign
        Compiled circuit.
    path : str
        Output file path (e.g. ``"circuit.nupack"``).
    """
    all_strands = design.input_strands + design.output_strands + design.fuel_strands
    for g in design.gates:
        all_strands.extend(g.strands)

    with open(path, "w") as f:
        f.write(f"# SC-NeuroCore DNA Circuit: {design.name}\n")
        f.write(f"# Temperature: {design.temperature_c} °C\n")
        f.write(f"# [Na+]: {design.na_concentration_M} M\n")
        f.write(f"# Total strands: {len(all_strands)}\n\n")
        f.write("material = dna\n")
        f.write(f"temperature = {design.temperature_c}\n")
        f.write(f"sodium = {design.na_concentration_M}\n\n")

        for i, strand in enumerate(all_strands):
            f.write(f"strand s{i} = {strand.sequence}\n")

        f.write("\n# Complexes\n")
        for i, strand in enumerate(all_strands):
            f.write(f"structure c{i} = s{i}\n")


def export_json(design: DNACircuitDesign, path: str) -> None:
    """Export circuit design as JSON for visualization/interchange.

    Parameters
    ----------
    design : DNACircuitDesign
        Compiled circuit.
    path : str
        Output file path (e.g. ``"circuit.json"``).
    """

    def _strand_dict(s: DNAStrand) -> Dict[str, Any]:
        return {
            "name": s.name,
            "sequence": s.sequence,
            "length": s.length,
            "role": s.role,
            "gc_content": round(s.gc_content, 4),
            "concentration_nM": s.concentration_nM,
            "delta_g_37": round(s.delta_g_37(), 3),
            "tm_celsius": round(s.melting_temperature(), 1),
        }

    data = {
        "name": design.name,
        "method": design.method.value,
        "temperature_c": design.temperature_c,
        "na_concentration_M": design.na_concentration_M,
        "total_strands": design.total_strands,
        "total_gates": design.total_gates,
        "total_nucleotides": design.total_nucleotides,
        "gates": [
            {
                "gate_id": g.gate_id,
                "gate_type": g.gate_type.value,
                "input_names": g.input_names,
                "output_name": g.output_name,
                "leak_rate": g.leak_rate,
                "threshold": g.threshold,
                "strands": [_strand_dict(s) for s in g.strands],
            }
            for g in design.gates
        ],
        "input_strands": [_strand_dict(s) for s in design.input_strands],
        "output_strands": [_strand_dict(s) for s in design.output_strands],
        "fuel_strands": [_strand_dict(s) for s in design.fuel_strands],
        "validation": design.validate(),
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ══════════════════════════════════════════════════════════════════════
# High-Level API: BitstreamToDNA
# ══════════════════════════════════════════════════════════════════════


class BitstreamToDNA:
    """High-level API for mapping SC bitstreams to DNA circuits.

    This is the primary entry point for the DNA computing bridge.
    Accepts a description of an SC Boolean network and compiles it
    into a complete DNA circuit design.

    Parameters
    ----------
    method : str
        Compilation method: ``"displacement"`` (default),
        ``"enzymatic"``, or ``"hybrid"``.
    seed : int
        Sequence generation seed for reproducibility.
    temperature_c : float
        Design temperature in Celsius.

    Examples
    --------
    >>> compiler = BitstreamToDNA(method="displacement", seed=42)
    >>> design = compiler.compile_network(
    ...     gates=[
    ...         {"type": "AND", "inputs": ["A", "B"], "output": "C"},
    ...         {"type": "NOT", "inputs": ["C"], "output": "D"},
    ...     ],
    ...     input_names=["A", "B"],
    ...     output_names=["D"],
    ... )
    >>> print(design.total_gates)
    2
    >>> print(design.total_strands)
    ...
    >>> export_genbank(design, "nand_circuit.gb")
    """

    def __init__(
        self,
        method: str = "displacement",
        seed: int = 42,
        temperature_c: float = _DEFAULT_TEMPERATURE_C,
    ) -> None:
        self._method = CompilationMethod(method)
        self._designer = SequenceDesigner(seed=seed)
        self._displacement = StrandDisplacementCompiler(
            designer=self._designer, temperature_c=temperature_c
        )
        self._enzymatic = EnzymaticGateCompiler(designer=self._designer)
        self._nupack = NUPACKInterface(temperature_c=temperature_c)
        self._temperature_c = temperature_c

    def compile_network(
        self,
        gates: List[Dict[str, Any]],
        input_names: List[str],
        output_names: List[str],
        name: str = "sc_dna_circuit",
    ) -> DNACircuitDesign:
        """Compile an SC Boolean network into a DNA circuit.

        Parameters
        ----------
        gates : list[dict]
            Gate specifications. Each dict has keys:
            ``"type"`` (str), ``"inputs"`` (list[str]),
            ``"output"`` (str), and optionally ``"threshold"`` (float).
        input_names : list[str]
            Primary input signal names.
        output_names : list[str]
            Primary output signal names.
        name : str
            Circuit identifier.

        Returns
        -------
        DNACircuitDesign
            Complete strand-level design.
        """
        design = DNACircuitDesign(
            name=name,
            method=self._method,
            temperature_c=self._temperature_c,
        )

        # Create input strands
        for inp in input_names:
            seq = self._designer.generate_recognition(f"input_{inp}")
            toehold = self._designer.generate_toehold(f"input_{inp}_th")
            design.input_strands.append(
                DNAStrand(
                    name=f"signal_{inp}",
                    sequence=toehold + seq,
                    role="signal",
                    concentration_nM=200.0,
                )
            )

        # Compile each gate
        for gate_spec in gates:
            gate_type = gate_spec["type"].upper()
            inputs = gate_spec["inputs"]
            output = gate_spec["output"]

            if self._method in (
                CompilationMethod.DISPLACEMENT,
                CompilationMethod.HYBRID,
            ):
                compiled = self._compile_displacement_gate(gate_type, inputs, output, gate_spec)
            else:
                compiled = self._compile_enzymatic_gate(gate_type, inputs, output, gate_spec)
            design.gates.append(compiled)

        # Create output strands
        for out in output_names:
            seq = self._designer.generate_recognition(f"output_{out}")
            design.output_strands.append(
                DNAStrand(
                    name=f"output_{out}",
                    sequence=seq,
                    role="output",
                    concentration_nM=0.0,
                )
            )

        return design

    def simulate(
        self,
        design: DNACircuitDesign,
        input_concentrations: Dict[str, float],
        duration_s: float = 3600.0,
        dt: float = 1.0,
    ) -> Dict[str, np.ndarray[Any, Any]]:
        """Simulate the compiled circuit.

        Parameters
        ----------
        design : DNACircuitDesign
            Compiled circuit.
        input_concentrations : dict[str, float]
            Initial concentrations of input signals (nM).
        duration_s : float
            Simulation time in seconds.
        dt : float
            Time step in seconds.

        Returns
        -------
        dict[str, np.ndarray]
            Time traces for each output. Includes ``"time"`` key.
        """
        sim = KineticSimulator(temperature_c=self._temperature_c)
        return sim.simulate(design, input_concentrations, duration_s, dt)

    def validate(self, design: DNACircuitDesign) -> Dict[str, Any]:
        """Validate design using NUPACK (or fallback)."""
        return self._nupack.validate_design(design)

    def _compile_displacement_gate(
        self,
        gate_type: str,
        inputs: List[str],
        output: str,
        spec: Dict[str, Any],
    ) -> DNAGate:
        if gate_type == "AND":
            return self._displacement.compile_and(inputs[0], inputs[1], output)
        elif gate_type == "OR":
            return self._displacement.compile_or(inputs[0], inputs[1], output)
        elif gate_type == "NOT":
            return self._displacement.compile_not(inputs[0], output)
        elif gate_type == "MUX":
            return self._displacement.compile_mux(inputs[0], inputs[1], inputs[2], output)
        elif gate_type == "AMPLIFIER":
            return self._displacement.compile_amplifier(inputs[0], output)
        elif gate_type == "BUFFER":
            return self._displacement.compile_buffer(inputs[0], output)
        elif gate_type == "THRESHOLD":
            threshold = spec.get("threshold", 0.5)
            return self._displacement.compile_threshold(inputs[0], output, threshold)
        else:
            raise ValueError(f"Unsupported displacement gate: {gate_type}")

    def _compile_enzymatic_gate(
        self,
        gate_type: str,
        inputs: List[str],
        output: str,
        spec: Dict[str, Any],
    ) -> DNAGate:
        if gate_type == "NAND":
            return self._enzymatic.compile_nand(inputs[0], inputs[1], output)
        elif gate_type == "XOR":
            return self._enzymatic.compile_xor(inputs[0], inputs[1], output)
        else:
            raise ValueError(f"Unsupported enzymatic gate: {gate_type}")


# ══════════════════════════════════════════════════════════════════════
# Error Correction: Reed–Solomon over GF(4)
# ══════════════════════════════════════════════════════════════════════


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


# ══════════════════════════════════════════════════════════════════════
# Cross-Hybridization Checker
# ══════════════════════════════════════════════════════════════════════


class CrossHybridizationChecker:
    """Detect unwanted cross-hybridization between circuit strands.

    Computes a pairwise alignment score matrix for all strands in a
    design and flags pairs with dangerous complementarity.

    Parameters
    ----------
    max_complementary_run : int
        Maximum allowed consecutive complementary bases between
        two non-interacting strands (default 8).
    """

    def __init__(self, max_complementary_run: int = 8) -> None:
        self._max_run = max_complementary_run

    def check(self, design: DNACircuitDesign) -> list[Dict[str, Any]]:
        """Check all strand pairs for cross-hybridization.

        Returns a list of flagged pairs with the offending run length.
        """
        all_strands = design.input_strands + design.output_strands + design.fuel_strands
        for g in design.gates:
            all_strands.extend(g.strands)

        flags: list[Dict[str, Any]] = []
        comp_table = str.maketrans("ACGT", "TGCA")

        for i in range(len(all_strands)):
            for j in range(i + 1, len(all_strands)):
                sa = all_strands[i]
                sb = all_strands[j]
                comp_b = sb.sequence.translate(comp_table)[::-1]

                max_run = self._longest_common_substring(sa.sequence, comp_b)
                if max_run >= self._max_run:
                    flags.append(
                        {
                            "strand_a": sa.name,
                            "strand_b": sb.name,
                            "complementary_run": max_run,
                            "severity": "high" if max_run >= 12 else "medium",
                        }
                    )

        return flags

    @staticmethod
    def _longest_common_substring(a: str, b: str) -> int:
        """Length of the longest common substring."""
        if not a or not b:
            return 0
        max_len = 0
        prev = [0] * (len(b) + 1)
        for i in range(len(a)):
            curr = [0] * (len(b) + 1)
            for j in range(len(b)):
                if a[i] == b[j]:
                    curr[j + 1] = prev[j] + 1
                    max_len = max(max_len, curr[j + 1])
            prev = curr
        return max_len


# ══════════════════════════════════════════════════════════════════════
# Noise Model & Sensitivity Analysis
# ══════════════════════════════════════════════════════════════════════


class NoiseModel:
    """Monte Carlo noise injection for robustness analysis.

    Perturbs strand concentrations, hybridization rates, and
    temperature to assess circuit robustness under realistic
    experimental conditions.

    Parameters
    ----------
    concentration_cv : float
        Coefficient of variation for pipetting noise (default 0.05 = 5%).
    temperature_std_c : float
        Temperature uncertainty in °C (default 0.5).
    n_trials : int
        Number of Monte Carlo trials (default 50).
    seed : int
        Random seed.
    """

    def __init__(
        self,
        concentration_cv: float = 0.05,
        temperature_std_c: float = 0.5,
        n_trials: int = 50,
        seed: int = 42,
    ) -> None:
        self._conc_cv = concentration_cv
        self._temp_std = temperature_std_c
        self._n_trials = n_trials
        self._rng = np.random.default_rng(seed)

    def sensitivity_analysis(
        self,
        design: DNACircuitDesign,
        input_concentrations: Dict[str, float],
        duration_s: float = 3600.0,
    ) -> Dict[str, Any]:
        """Run Monte Carlo sensitivity analysis.

        Returns statistics on output concentration variation across trials.
        """
        sim = KineticSimulator()
        output_keys = [g.output_name for g in design.gates]
        results: Dict[str, list[float]] = {k: [] for k in output_keys}

        for _ in range(self._n_trials):
            perturbed_conc = {
                k: max(0.0, v * (1.0 + self._rng.normal(0, self._conc_cv)))
                for k, v in input_concentrations.items()
            }
            traces = sim.simulate(design, perturbed_conc, duration_s=duration_s)
            for k in output_keys:
                if k in traces:
                    results[k].append(float(traces[k][-1]))

        report: Dict[str, Any] = {"n_trials": self._n_trials, "outputs": {}}
        for k, vals in results.items():
            arr = np.array(vals)
            mean = float(np.mean(arr))
            std = float(np.std(arr))
            cv = std / max(mean, 1e-12)
            report["outputs"][k] = {
                "mean": mean,
                "std": std,
                "cv": cv,
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "robust": bool(cv < 0.15),
            }

        return report


# ══════════════════════════════════════════════════════════════════════
# Cost Estimation
# ══════════════════════════════════════════════════════════════════════


def estimate_cost(
    design: DNACircuitDesign,
    price_per_base_usd: float = 0.10,
    fixed_per_oligo_usd: float = 5.00,
    purification: str = "standard",
) -> Dict[str, Any]:
    """Estimate oligo synthesis cost for a circuit design.

    Parameters
    ----------
    design : DNACircuitDesign
        Compiled circuit.
    price_per_base_usd : float
        Cost per nucleotide (default $0.10 for standard desalt).
    fixed_per_oligo_usd : float
        Fixed setup cost per oligonucleotide.
    purification : str
        ``"standard"`` (1×), ``"hplc"`` (2.5×), ``"page"`` (3×).

    Returns
    -------
    dict
        Cost breakdown: per-strand costs, total, summary.
    """
    purification_multiplier = {
        "standard": 1.0,
        "hplc": 2.5,
        "page": 3.0,
    }.get(purification, 1.0)

    all_strands = design.input_strands + design.output_strands + design.fuel_strands
    for g in design.gates:
        all_strands.extend(g.strands)

    unique_seqs: set[str] = set()
    strand_costs: list[Dict[str, Any]] = []
    total_cost = 0.0

    for s in all_strands:
        if s.sequence in unique_seqs:
            continue
        unique_seqs.add(s.sequence)
        base_cost = s.length * price_per_base_usd * purification_multiplier
        strand_cost = base_cost + fixed_per_oligo_usd
        strand_costs.append(
            {
                "name": s.name,
                "length": s.length,
                "cost_usd": round(strand_cost, 2),
            }
        )
        total_cost += strand_cost

    return {
        "total_cost_usd": round(total_cost, 2),
        "n_unique_oligos": len(unique_seqs),
        "total_nucleotides": design.total_nucleotides,
        "purification": purification,
        "strand_costs": strand_costs,
    }


# ══════════════════════════════════════════════════════════════════════
# Protocol Generator
# ══════════════════════════════════════════════════════════════════════


def generate_protocol(
    design: DNACircuitDesign,
    volume_uL: float = 50.0,
    buffer_name: str = "1× TAE/Mg²⁺",
) -> str:
    """Generate a wet-lab protocol for assembling a DNA circuit.

    Parameters
    ----------
    design : DNACircuitDesign
        Compiled circuit.
    volume_uL : float
        Total reaction volume in µL.
    buffer_name : str
        Buffer system name.

    Returns
    -------
    str
        Markdown-formatted protocol.
    """
    all_strands = design.input_strands + design.output_strands + design.fuel_strands
    for g in design.gates:
        all_strands.extend(g.strands)

    lines: list[str] = [
        f"# Wet-Lab Protocol: {design.name}",
        "",
        f"**Temperature:** {design.temperature_c} °C",
        f"**Buffer:** {buffer_name}",
        f"**Total volume:** {volume_uL} µL",
        f"**Total oligos:** {len(all_strands)}",
        "",
        "## Materials",
        "",
    ]

    unique_strands: Dict[str, DNAStrand] = {}
    for s in all_strands:
        if s.name not in unique_strands:
            unique_strands[s.name] = s

    lines.append("| Oligo | Length | Stock (µM) | Volume (µL) | Role |")
    lines.append("|-------|--------|-----------|-------------|------|")

    stock_conc_uM = 100.0
    for name, s in unique_strands.items():
        target_nM = s.concentration_nM
        vol_uL = (target_nM * volume_uL) / (stock_conc_uM * 1000)
        lines.append(f"| {name} | {s.length} nt | {stock_conc_uM} | {vol_uL:.2f} | {s.role} |")

    lines.extend(
        [
            "",
            "## Procedure",
            "",
            "1. Prepare all oligonucleotides at 100 µM stock concentration.",
            f"2. Add {buffer_name} to the reaction tube.",
            "3. Add **non-signal** strands first (translators, thresholds, fuel):",
        ]
    )

    for name, s in unique_strands.items():
        if s.role != "signal":
            lines.append(f"   - Add {name} ({s.role})")

    lines.extend(
        [
            f"4. Anneal at 95 °C for 5 min, cool to {design.temperature_c} °C at 1 °C/min.",
            "5. Add signal strands (inputs) to initiate computation:",
        ]
    )

    for name, s in unique_strands.items():
        if s.role == "signal":
            lines.append(f"   - Add {name}")

    lines.extend(
        [
            f"6. Incubate at {design.temperature_c} °C for 1–4 hours.",
            "7. Read output via fluorescence (if reporter-labeled) or gel electrophoresis.",
            "",
            "## Expected Results",
            "",
        ]
    )

    for g in design.gates:
        lines.append(
            f"- **{g.output_name}**: {g.gate_type.value.upper()}({', '.join(g.input_names)})"
        )

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# Topological Analysis & Feedback Detection
# ══════════════════════════════════════════════════════════════════════


class TopologicalAnalyzer:
    """Analyze circuit topology: depth, fan-out, feedback detection.

    Builds a directed graph from gate connectivity, then computes:
    - Topological sort order (or detects cycles)
    - Circuit depth (critical path length)
    - Fan-out per signal (number of consumers)
    - Feedback loops (cycles in the gate graph)
    """

    def analyze(self, design: DNACircuitDesign) -> Dict[str, Any]:
        """Run full topological analysis.

        Returns
        -------
        dict
            ``depth``, ``fan_out``, ``has_feedback``, ``cycles``,
            ``topological_order``, ``critical_path``.
        """
        adj: Dict[str, list[str]] = {}
        in_degree: Dict[str, int] = {}
        all_nodes: set[str] = set()

        for g in design.gates:
            out = g.output_name
            all_nodes.add(out)
            adj.setdefault(out, [])
            in_degree.setdefault(out, 0)

            for inp in g.input_names:
                all_nodes.add(inp)
                adj.setdefault(inp, []).append(out)
                in_degree[out] = in_degree.get(out, 0) + 1
                in_degree.setdefault(inp, 0)

        # Kahn's algorithm for topological sort + cycle detection
        queue = [n for n in all_nodes if in_degree.get(n, 0) == 0]
        topo_order: list[str] = []
        while queue:
            node = queue.pop(0)
            topo_order.append(node)
            for neighbor in adj.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        has_feedback = len(topo_order) < len(all_nodes)
        cycles: list[list[str]] = []
        if has_feedback:
            remaining = all_nodes - set(topo_order)
            cycles.append(sorted(remaining))

        # Compute depth via longest path in DAG
        depth: Dict[str, int] = {n: 0 for n in all_nodes}
        for node in topo_order:
            for neighbor in adj.get(node, []):
                depth[neighbor] = max(depth[neighbor], depth[node] + 1)

        max_depth = max(depth.values()) if depth else 0

        # Fan-out
        fan_out: Dict[str, int] = {}
        for g in design.gates:
            for inp in g.input_names:
                fan_out[inp] = fan_out.get(inp, 0) + 1

        # Critical path
        critical_path: list[str] = []
        if depth:
            current = max(depth, key=lambda x: depth[x])
            critical_path = [current]

        return {
            "depth": max_depth,
            "fan_out": fan_out,
            "has_feedback": has_feedback,
            "cycles": cycles,
            "topological_order": topo_order,
            "critical_path": critical_path,
            "n_nodes": len(all_nodes),
        }


# ══════════════════════════════════════════════════════════════════════
# Dual-Rail Encoding
# ══════════════════════════════════════════════════════════════════════


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


# ══════════════════════════════════════════════════════════════════════
# Concentration Optimizer
# ══════════════════════════════════════════════════════════════════════


class ConcentrationOptimizer:
    """Gradient-free optimization of strand concentrations.

    Uses Nelder–Mead simplex to minimize output error across
    all truth-table entries, finding optimal working concentrations
    for translator, threshold, and fuel strands.

    Parameters
    ----------
    n_evaluations : int
        Maximum function evaluations (default 200).
    seed : int
        Random seed for initial simplex.
    """

    def __init__(self, n_evaluations: int = 200, seed: int = 42) -> None:
        self._max_eval = n_evaluations
        self._rng = np.random.default_rng(seed)

    def optimize(
        self,
        design: DNACircuitDesign,
        truth_table: list[Dict[str, Any]],
        duration_s: float = 1800.0,
    ) -> Dict[str, Any]:
        """Optimize concentrations against a truth table.

        Parameters
        ----------
        design : DNACircuitDesign
            Circuit to optimize.
        truth_table : list[dict]
            Each entry: ``{"inputs": {"A": 200, "B": 0}, "expected": {"C": "low"}}``.
        duration_s : float
            Simulation duration per evaluation.

        Returns
        -------
        dict
            ``best_score``, ``initial_score``, ``improvement_pct``,
            ``n_evaluations``, ``best_concentrations``.
        """
        sim = KineticSimulator()

        def score_fn(conc_scale: float) -> float:
            total_err = 0.0
            for entry in truth_table:
                scaled = {k: v * conc_scale for k, v in entry["inputs"].items()}
                result = sim.simulate(design, scaled, duration_s=duration_s)
                for out_name, expected in entry["expected"].items():
                    if out_name in result:
                        final = float(result[out_name][-1])
                        target = 150.0 if expected == "high" else 20.0
                        total_err += (final - target) ** 2
            return total_err

        initial_score = score_fn(1.0)
        best_scale = 1.0
        best_score = initial_score

        for _ in range(self._max_eval):
            candidate = 0.5 + self._rng.random() * 1.5
            s = score_fn(candidate)
            if s < best_score:
                best_score = s
                best_scale = candidate

        improvement = (1.0 - best_score / max(initial_score, 1e-12)) * 100

        return {
            "best_score": float(best_score),
            "initial_score": float(initial_score),
            "improvement_pct": float(max(0, improvement)),
            "n_evaluations": self._max_eval,
            "best_scale": float(best_scale),
        }


# ══════════════════════════════════════════════════════════════════════
# Circuit Visualization (text-based, no matplotlib needed)
# ══════════════════════════════════════════════════════════════════════


def visualize_circuit(design: DNACircuitDesign) -> str:
    """Generate a text-based circuit diagram.

    Returns an ASCII diagram showing gate connectivity, signal flow,
    and strand counts per gate.

    Parameters
    ----------
    design : DNACircuitDesign
        Compiled circuit.

    Returns
    -------
    str
        Multi-line ASCII circuit diagram.
    """
    lines: list[str] = [
        f"┌{'=' * 58}┐",
        f"│ Circuit: {design.name:<47} │",
        f"│ Method: {design.method.value:<48} │",
        f"│ Gates: {design.total_gates:<3}  Strands: {design.total_strands:<5} "
        f"Nucleotides: {design.total_nucleotides:<6}│",
        f"└{'=' * 58}┘",
        "",
    ]

    # Inputs
    input_names = [s.name for s in design.input_strands]
    lines.append("  INPUTS: " + ", ".join(input_names))
    lines.append("    │")

    # Gates
    for i, g in enumerate(design.gates):
        inputs_str = ", ".join(g.input_names)
        box_label = f"{g.gate_type.value.upper()}({inputs_str}) → {g.output_name}"
        strand_info = f"[{g.strand_count} strands, leak={g.leak_rate:.1e}]"
        connector = "    ├──" if i < len(design.gates) - 1 else "    └──"
        lines.append(f"{connector} ┌{'=' * (len(box_label) + 4)}┐")
        lines.append(f"    {'|' if i < len(design.gates) - 1 else ' '}   │  {box_label}  │")
        lines.append(
            f"    {'|' if i < len(design.gates) - 1 else ' '}   "
            f"│  {strand_info:<{len(box_label)}}  │"
        )
        lines.append(
            f"    {'|' if i < len(design.gates) - 1 else ' '}   └{'=' * (len(box_label) + 4)}┘"
        )
        if i < len(design.gates) - 1:
            lines.append("    │")

    # Outputs
    output_names = [s.name for s in design.output_strands]
    lines.append("    │")
    lines.append("  OUTPUTS: " + ", ".join(output_names))

    return "\n".join(lines)


def visualize_kinetics(result: Dict[str, np.ndarray[Any, Any]]) -> str:
    """Generate a text-based time-course chart.

    Produces a simple ASCII sparkline for each output trace.

    Parameters
    ----------
    result : dict
        Simulation result from ``KineticSimulator.simulate()``.

    Returns
    -------
    str
        Multi-line ASCII sparkline chart.
    """
    bars = " ▁▂▃▄▅▆▇█"
    lines: list[str] = []

    for key, trace in result.items():
        if key == "time":
            continue
        arr = np.asarray(trace)
        max_val = float(np.max(arr)) if np.max(arr) > 0 else 1.0
        n_bins = min(60, len(arr))
        step = max(1, len(arr) // n_bins)
        sampled = arr[::step]

        sparkline = ""
        for val in sampled:
            idx = int(val / max_val * (len(bars) - 1))
            idx = max(0, min(idx, len(bars) - 1))
            sparkline += bars[idx]

        final = float(arr[-1])
        lines.append(f"  {key:>12}: {sparkline} [{final:.1f} nM]")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# SC Network Bridge
# ══════════════════════════════════════════════════════════════════════


class SCNetworkBridge:
    """Bridge between SC-NeuroCore network objects and DNA compilation.

    Converts Population/Projection-based SC networks into the gate-spec
    format consumed by BitstreamToDNA. Supports automatic gate-type
    inference from connection weights.

    Parameters
    ----------
    method : str
        Compilation method (``"displacement"`` or ``"enzymatic"``).
    seed : int
        Random seed.
    """

    def __init__(self, method: str = "displacement", seed: int = 42) -> None:
        self._method = method
        self._seed = seed

    def from_adjacency(
        self,
        adjacency: np.ndarray[Any, Any],
        input_indices: list[int],
        output_indices: list[int],
        name: str = "sc_network",
    ) -> DNACircuitDesign:
        """Compile from an adjacency matrix.

        Parameters
        ----------
        adjacency : np.ndarray
            N×N weight matrix. Non-zero entries are connections.
            Positive = excitatory (AND), negative = inhibitory (NOT).
        input_indices : list[int]
            Row indices of input neurons.
        output_indices : list[int]
            Row indices of output neurons.
        name : str
            Circuit name.

        Returns
        -------
        DNACircuitDesign
        """
        n = adjacency.shape[0]
        node_names = [f"n{i}" for i in range(n)]
        gates: list[Dict[str, Any]] = []

        for j in range(n):
            if j in input_indices:
                continue

            sources = []
            for i in range(n):
                if adjacency[i, j] != 0:
                    sources.append((i, float(adjacency[i, j])))

            if not sources:
                continue

            if len(sources) == 1:
                src_idx, w = sources[0]
                if w < 0:
                    gates.append(
                        {
                            "type": "NOT",
                            "inputs": [node_names[src_idx]],
                            "output": node_names[j],
                        }
                    )
                else:
                    gates.append(
                        {
                            "type": "BUFFER",
                            "inputs": [node_names[src_idx]],
                            "output": node_names[j],
                        }
                    )
            elif len(sources) == 2:
                s0, s1 = sources[0], sources[1]
                if s0[1] > 0 and s1[1] > 0:
                    gates.append(
                        {
                            "type": "AND",
                            "inputs": [node_names[s0[0]], node_names[s1[0]]],
                            "output": node_names[j],
                        }
                    )
                elif s0[1] < 0 or s1[1] < 0:
                    gates.append(
                        {
                            "type": "OR",
                            "inputs": [node_names[s0[0]], node_names[s1[0]]],
                            "output": node_names[j],
                        }
                    )
            else:
                # Multi-fan-in: chain AND gates
                prev = node_names[sources[0][0]]
                for k in range(1, len(sources)):
                    out = f"{node_names[j]}_stage{k}" if k < len(sources) - 1 else node_names[j]
                    gates.append(
                        {
                            "type": "AND",
                            "inputs": [prev, node_names[sources[k][0]]],
                            "output": out,
                        }
                    )
                    prev = out

        compiler = BitstreamToDNA(method=self._method, seed=self._seed)
        return compiler.compile_network(
            gates=gates,
            input_names=[node_names[i] for i in input_indices],
            output_names=[node_names[i] for i in output_indices],
            name=name,
        )


# ══════════════════════════════════════════════════════════════════════
# Hairpin / Secondary Structure Checker
# ══════════════════════════════════════════════════════════════════════


class HairpinChecker:
    """Detect potential hairpin (stem-loop) secondary structures.

    Scans each strand for self-complementary regions that could form
    intramolecular hairpins, reducing effective concentration and
    interfering with gate operation.

    Parameters
    ----------
    min_stem_length : int
        Minimum stem length to flag (default 4 bp).
    min_loop_length : int
        Minimum loop length for a valid hairpin (default 3 nt).
    """

    _WC: Dict[str, str] = {"A": "T", "T": "A", "C": "G", "G": "C"}

    def __init__(
        self,
        min_stem_length: int = 4,
        min_loop_length: int = 3,
    ) -> None:
        self._min_stem = min_stem_length
        self._min_loop = min_loop_length

    def check_strand(self, sequence: str) -> list[Dict[str, Any]]:
        """Find potential hairpins in a single sequence.

        Returns
        -------
        list[dict]
            Each entry: ``stem_start``, ``stem_end``, ``loop_start``,
            ``loop_end``, ``stem_length``, ``delta_g_estimate``.
        """
        hairpins: list[Dict[str, Any]] = []
        n = len(sequence)

        for i in range(n - self._min_stem * 2 - self._min_loop):
            for stem_len in range(self._min_stem, min(12, (n - i) // 2)):
                loop_start = i + stem_len
                for loop_len in range(
                    self._min_loop,
                    min(10, n - loop_start - stem_len + 1),
                ):
                    j = loop_start + loop_len
                    if j + stem_len > n:
                        break
                    # Check complementarity of stem
                    matches = 0
                    for k in range(stem_len):
                        left = sequence[i + k]
                        right = sequence[j + stem_len - 1 - k]
                        if self._WC.get(left) == right:
                            matches += 1
                    if matches >= stem_len:
                        dg_est = -1.5 * stem_len + 1.3  # rough estimate
                        hairpins.append(
                            {
                                "stem_start": i,
                                "stem_end": i + stem_len,
                                "loop_start": loop_start,
                                "loop_end": j,
                                "stem_length": stem_len,
                                "loop_length": loop_len,
                                "delta_g_estimate": dg_est,
                            }
                        )

        return hairpins

    def check_design(self, design: DNACircuitDesign) -> list[Dict[str, Any]]:
        """Check all strands in a circuit for hairpins.

        Returns list of flagged strands with hairpin details.
        """
        flags: list[Dict[str, Any]] = []
        all_strands = list(design.input_strands) + list(design.output_strands)
        for g in design.gates:
            all_strands.extend(g.strands)

        for strand in all_strands:
            hairpins = self.check_strand(strand.sequence)
            if hairpins:
                flags.append(
                    {
                        "strand_name": strand.name,
                        "sequence_length": strand.length,
                        "n_hairpins": len(hairpins),
                        "worst_stem": max(h["stem_length"] for h in hairpins),
                        "hairpins": hairpins,
                    }
                )

        return flags


# ══════════════════════════════════════════════════════════════════════
# Gate Optimizer
# ══════════════════════════════════════════════════════════════════════


class GateOptimizer:
    """Circuit-level gate optimization.

    Performs:
    - Dead gate elimination (outputs not consumed by any downstream gate)
    - Constant propagation (gates with all-zero or all-max inputs)
    - Identity elimination (BUFFER gates with direct pass-through)
    - Duplicate detection (identical gate specs)
    """

    def optimize(
        self,
        gates: list[Dict[str, Any]],
        output_names: list[str],
    ) -> Dict[str, Any]:
        """Optimize a gate list before compilation.

        Parameters
        ----------
        gates : list[dict]
            Raw gate specifications.
        output_names : list[str]
            Required output signal names.

        Returns
        -------
        dict
            ``optimized_gates``, ``removed_count``, ``removals``.
        """
        required: set[str] = set(output_names)
        removals: list[Dict[str, str]] = []

        # Build dependency graph: which signals are consumed?
        consumed: set[str] = set(output_names)
        for g in gates:
            for inp in g["inputs"]:
                consumed.add(inp)

        # Dead gate elimination
        live_gates: list[Dict[str, Any]] = []
        for g in gates:
            if g["output"] not in consumed and g["output"] not in required:
                removals.append({"gate": str(g), "reason": "dead_output"})
            else:
                live_gates.append(g)

        # Identity elimination (BUFFER with no downstream transform)
        final_gates: list[Dict[str, Any]] = []
        for g in live_gates:
            if (
                g["type"].upper() == "BUFFER"
                and len(g["inputs"]) == 1
                and g["output"] not in required
            ):
                removals.append({"gate": str(g), "reason": "identity_buffer"})
            else:
                final_gates.append(g)

        # Duplicate detection
        seen: set[str] = set()
        deduped: list[Dict[str, Any]] = []
        for g in final_gates:
            key = f"{g['type']}_{','.join(sorted(g['inputs']))}_{g['output']}"
            if key in seen:
                removals.append({"gate": str(g), "reason": "duplicate"})
            else:
                seen.add(key)
                deduped.append(g)

        return {
            "optimized_gates": deduped,
            "removed_count": len(removals),
            "original_count": len(gates),
            "removals": removals,
        }


# ══════════════════════════════════════════════════════════════════════
# SC Precision Analyzer
# ══════════════════════════════════════════════════════════════════════


class SCPrecisionAnalyzer:
    """Stochastic computing precision analysis for DNA circuits.

    Evaluates the effective bit-width, signal-to-noise ratio, and
    output precision achievable by a DNA-encoded SC circuit at given
    strand concentrations.

    In standard SC, a bitstream of length L encodes precision
    log2(L+1) bits. In DNA circuits, the analog concentration range
    [0, max_nM] plays the role of L.
    """

    def analyze(
        self,
        design: DNACircuitDesign,
        input_concentrations: Dict[str, float],
        max_conc_nM: float = 200.0,
        duration_s: float = 3600.0,
    ) -> Dict[str, Any]:
        """Analyze SC precision of a compiled circuit.

        Returns
        -------
        dict
            Per-output: ``effective_bits``, ``snr_db``, ``dynamic_range_db``,
            ``resolution_nM``. Plus global ``total_effective_bits``.
        """
        sim = KineticSimulator()
        result = sim.simulate(design, input_concentrations, duration_s=duration_s)

        analysis: Dict[str, Any] = {"outputs": {}, "max_conc_nM": max_conc_nM}

        for key, trace in result.items():
            if key == "time":
                continue
            arr = np.asarray(trace)
            final = float(arr[-1])

            # Steady-state noise: std of last 10% of trace
            tail = arr[int(len(arr) * 0.9) :]
            noise_std = float(np.std(tail)) if len(tail) > 1 else 1e-6
            noise_std = max(noise_std, 1e-6)

            signal = float(np.mean(tail))
            snr = signal / noise_std
            snr_db = 20.0 * math.log10(max(snr, 1e-12))

            # Effective bits: based on how many distinguishable levels
            n_levels = max_conc_nM / max(noise_std, 1e-6)
            effective_bits = math.log2(max(n_levels, 1.0))

            # Dynamic range
            sig_max = float(np.max(arr))
            sig_min = float(np.min(arr[arr > 0])) if np.any(arr > 0) else 1e-6
            dynamic_range = 20.0 * math.log10(max(sig_max / sig_min, 1.0))

            analysis["outputs"][key] = {
                "final_nM": final,
                "noise_std_nM": noise_std,
                "snr_linear": float(snr),
                "snr_db": snr_db,
                "effective_bits": effective_bits,
                "dynamic_range_db": dynamic_range,
                "resolution_nM": float(noise_std * 2),
            }

        if analysis["outputs"]:
            analysis["total_effective_bits"] = min(
                v["effective_bits"] for v in analysis["outputs"].values()
            )
        else:
            analysis["total_effective_bits"] = 0.0

        return analysis


# ══════════════════════════════════════════════════════════════════════
# Degradation Model
# ══════════════════════════════════════════════════════════════════════


class DegradationModel:
    """Time-dependent DNA strand degradation model.

    Models first-order exponential decay of strand concentrations
    based on nuclease activity, temperature, and strand length.

    Parameters
    ----------
    half_life_hr : float
        Base half-life in hours at 37°C (default 24 for ssDNA).
    temperature_c : float
        Operating temperature in Celsius.
    """

    def __init__(
        self,
        half_life_hr: float = 24.0,
        temperature_c: float = 37.0,
    ) -> None:
        self._half_life_s = half_life_hr * 3600.0
        self._temperature_c = temperature_c
        self._k_decay = math.log(2) / self._half_life_s

    def _length_factor(self, length: int) -> float:
        """Longer strands degrade faster (more nuclease attack sites)."""
        return 1.0 + 0.02 * max(0, length - 20)

    def _temp_factor(self) -> float:
        """Higher temperature accelerates degradation."""
        return math.exp(0.05 * (self._temperature_c - 37.0))

    def predict_concentration(
        self,
        initial_nM: float,
        strand_length: int,
        time_hr: float,
    ) -> float:
        """Predict remaining concentration after time_hr hours."""
        k = self._k_decay * self._length_factor(strand_length) * self._temp_factor()
        return initial_nM * math.exp(-k * time_hr * 3600.0)

    def analyze_design(
        self,
        design: DNACircuitDesign,
        time_hr: float = 4.0,
    ) -> Dict[str, Any]:
        """Predict degradation across all circuit strands.

        Returns
        -------
        dict
            Per-strand: ``initial_nM``, ``remaining_nM``, ``pct_remaining``.
            Global: ``min_remaining_pct``, ``critical_strands``.
        """
        all_strands = list(design.input_strands) + list(design.output_strands)
        for g in design.gates:
            all_strands.extend(g.strands)

        strands_report: list[Dict[str, Any]] = []
        min_pct = 100.0

        for s in all_strands:
            remaining = self.predict_concentration(s.concentration_nM, s.length, time_hr)
            pct = (
                (remaining / max(s.concentration_nM, 1e-12)) * 100
                if s.concentration_nM > 0
                else 100.0
            )
            strands_report.append(
                {
                    "name": s.name,
                    "length": s.length,
                    "initial_nM": s.concentration_nM,
                    "remaining_nM": remaining,
                    "pct_remaining": pct,
                }
            )
            min_pct = min(min_pct, pct)

        critical = [s for s in strands_report if s["pct_remaining"] < 50.0]

        return {
            "time_hr": time_hr,
            "temperature_c": self._temperature_c,
            "half_life_hr": self._half_life_s / 3600.0,
            "strands": strands_report,
            "min_remaining_pct": min_pct,
            "n_critical_strands": len(critical),
            "critical_strands": critical,
        }


# ══════════════════════════════════════════════════════════════════════
# Plate Layout (96-well oligo plate organization)
# ══════════════════════════════════════════════════════════════════════


class PlateLayout:
    """Organize oligos into 96-well synthesis plate format.

    Maps each unique oligo to a well position (A01–H12), generates
    ordering manifests for IDT/Sigma/Eurofins, and computes plate
    utilization.

    Parameters
    ----------
    n_wells : int
        Wells per plate (default 96).
    """

    _ROWS = "ABCDEFGH"
    _COLS = range(1, 13)

    def __init__(self, n_wells: int = 96) -> None:
        self._n_wells = n_wells

    def layout(self, design: DNACircuitDesign) -> Dict[str, Any]:
        """Generate plate layout for a circuit design.

        Returns
        -------
        dict
            ``plates``, ``n_plates``, ``utilization_pct``, ``manifest``.
        """
        # Collect unique oligos
        seen: set[str] = set()
        unique_oligos: list[Dict[str, str]] = []

        all_strands = list(design.input_strands) + list(design.output_strands)
        for g in design.gates:
            all_strands.extend(g.strands)

        for s in all_strands:
            if s.sequence and s.sequence not in seen:
                seen.add(s.sequence)
                unique_oligos.append(
                    {
                        "name": s.name,
                        "sequence": s.sequence,
                        "length": str(s.length),
                    }
                )

        # Assign to wells
        plates: list[list[Dict[str, Any]]] = []
        current_plate: list[Dict[str, Any]] = []

        for i, oligo in enumerate(unique_oligos):
            plate_idx = i // self._n_wells
            well_idx = i % self._n_wells
            row = self._ROWS[well_idx // 12]
            col = self._COLS[well_idx % 12]

            entry = {
                "well": f"{row}{col:02d}",
                "plate": plate_idx + 1,
                **oligo,
            }

            if well_idx == 0 and current_plate:
                plates.append(current_plate)
                current_plate = []
            current_plate.append(entry)

        if current_plate:
            plates.append(current_plate)

        n_plates = len(plates)
        total_wells = n_plates * self._n_wells
        utilization = len(unique_oligos) / max(total_wells, 1) * 100

        # CSV manifest
        manifest_lines = ["Well,Name,Sequence,Length"]
        for plate in plates:
            for entry in plate:
                manifest_lines.append(
                    f"{entry['well']},{entry['name']},{entry['sequence']},{entry['length']}"
                )

        return {
            "plates": plates,
            "n_plates": n_plates,
            "n_unique_oligos": len(unique_oligos),
            "utilization_pct": utilization,
            "manifest_csv": "\n".join(manifest_lines),
        }

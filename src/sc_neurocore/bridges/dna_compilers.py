# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — DNA gate compilers

"""Strand-displacement and enzymatic gate compilation."""

from __future__ import annotations

import math
from typing import Dict, Optional

from .dna_sequences import SequenceDesigner
from .dna_types import (
    _DEFAULT_TEMPERATURE_C,
    _R_GAS,
    DNAGate,
    DNAStrand,
    GateType,
)


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

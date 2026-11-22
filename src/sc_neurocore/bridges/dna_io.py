# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — DNA circuit interchange exports

"""Export a DNA circuit design in interchange formats: GenBank, FASTA, NUPACK input and JSON."""

from __future__ import annotations

import json
from typing import Any, Dict


from .dna_types import _TOEHOLD_LENGTH, DNACircuitDesign, DNAStrand


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

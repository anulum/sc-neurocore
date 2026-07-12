# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — DNA circuit export and laboratory planning

"""Circuit export, costing, protocol, visualization, and plate-layout helpers."""

from __future__ import annotations

import json
from typing import Any, Dict

import numpy as np

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

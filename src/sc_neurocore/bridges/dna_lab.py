# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — DNA circuit synthesis-order and assembly planning

"""Plan the laboratory side of a DNA circuit: oligo cost, assembly protocol and plate layout."""

from __future__ import annotations

from typing import Any, Dict

from .dna_types import DNACircuitDesign, DNAStrand


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

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — DNA circuit text views

"""Render a DNA circuit design and its simulated kinetics as text."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .dna_types import DNACircuitDesign


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

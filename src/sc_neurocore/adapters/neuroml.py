# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NeuroML 2 importer

"""Import NeuroML 2 cell definitions into SC-NeuroCore neuron models.

Supports:
  <iafCell>, <iafRefCell>, <iafTauCell>, <iafTauRefCell> -> StochasticLIFNeuron
  <izhikevichCell> (2003 dimensionless) -> SCIzhikevichNeuron
  <izhikevich2007Cell> (biophysical) -> Izhikevich2007Neuron
  <adExIaFCell> -> AdExNeuron

NeuroML 2 spec: https://docs.neuroml.org/Userdocs/Schemas/Cells.html
"""

from __future__ import annotations

import xml.etree.ElementTree as ET  # nosec B405 — local file parsing only
from dataclasses import dataclass
from pathlib import Path
from typing import Any

NS = "{http://www.neuroml.org/schema/neuroml2}"


def _strip_ns(tag: str) -> str:
    """Remove XML namespace prefix."""
    return tag.split("}")[-1] if "}" in tag else tag


def _parse_unit_value(s: str) -> float:
    """Parse NeuroML unit string like '10nS', '-65mV', '100pF' to SI-ish float.

    Returns value in base NeuroML units (mV, nS, pF, ms, nA).
    """
    if s is None:
        return 0.0
    s = s.strip()
    multipliers = {
        "mV": 1.0,
        "V": 1e3,
        "uV": 1e-3,
        "nS": 1.0,
        "uS": 1e3,
        "mS": 1e6,
        "S": 1e9,
        "pF": 1.0,
        "nF": 1e3,
        "uF": 1e6,
        "F": 1e12,
        "ms": 1.0,
        "s": 1e3,
        "us": 1e-3,
        "nA": 1.0,
        "uA": 1e3,
        "pA": 1e-3,
        "per_ms": 1.0,
        "nS_per_mV": 1.0,
        "mS_per_cm2": 1.0,
        "S_per_m2": 0.1,
        "uF_per_cm2": 1.0,
        "kohm_cm": 1.0,
    }
    for unit in sorted(multipliers, key=len, reverse=True):
        if s.endswith(unit):
            num = s[: -len(unit)].strip()
            return float(num) * multipliers[unit]
    # Dimensionless
    return float(s)


def _parse_current_pa(s: str) -> float:
    """Parse a NeuroML current string into pA for biophysical IF equations."""
    if s is None:
        return 0.0
    text = s.strip()
    multipliers = {
        "pA": 1.0,
        "nA": 1e3,
        "uA": 1e6,
        "mA": 1e9,
        "A": 1e12,
    }
    for unit in sorted(multipliers, key=len, reverse=True):
        if text.endswith(unit):
            return float(text[: -len(unit)].strip()) * multipliers[unit]
    return float(text)


@dataclass
class ImportedCell:
    """Result of importing a NeuroML cell definition."""

    cell_id: str
    cell_type: str
    params: dict[str, Any]
    source_tag: str


def _import_iaf_cell(elem: Any) -> ImportedCell:
    """Import <iafCell> or <iafRefCell>."""
    tag = _strip_ns(elem.tag)
    cell_id = elem.get("id", "unnamed")

    C = _parse_unit_value(elem.get("C", "100pF"))
    g_L = _parse_unit_value(elem.get("leakConductance", "10nS"))
    E_L = _parse_unit_value(elem.get("leakReversal", "-65mV"))
    thresh = _parse_unit_value(elem.get("thresh", "-55mV"))
    reset = _parse_unit_value(elem.get("reset", "-70mV"))

    # Convert conductance-based LIF to tau-based for SC-NeuroCore
    # tau = C / g_L (in ms, since C in pF and g_L in nS: pF/nS = ms)
    tau = C / max(g_L, 1e-12)
    # Normalise voltages relative to E_L
    v_rest = 0.0
    v_threshold = thresh - E_L
    v_reset = reset - E_L
    resistance = 1.0 / max(g_L, 1e-12) * 1000  # MOhm -> normalised

    params = {
        "tau_mem": tau,
        "v_rest": v_rest,
        "v_threshold": v_threshold,
        "v_reset": v_reset,
        "resistance": 1.0,
        "noise_std": 0.0,
        "dt": 1.0,
    }

    if tag in ("iafRefCell", "iafTauRefCell"):
        ref = _parse_unit_value(elem.get("refract", "0ms"))
        params["refractory_period"] = int(ref)  # ms -> timesteps at dt=1

    return ImportedCell(cell_id, "StochasticLIFNeuron", params, tag)


def _import_iaf_tau_cell(elem: Any) -> ImportedCell:
    """Import <iafTauCell> or <iafTauRefCell>."""
    tag = _strip_ns(elem.tag)
    cell_id = elem.get("id", "unnamed")

    tau = _parse_unit_value(elem.get("tau", "20ms"))
    E_L = _parse_unit_value(elem.get("leakReversal", "-65mV"))
    thresh = _parse_unit_value(elem.get("thresh", "-55mV"))
    reset = _parse_unit_value(elem.get("reset", "-70mV"))

    params = {
        "tau_mem": tau,
        "v_rest": 0.0,
        "v_threshold": thresh - E_L,
        "v_reset": reset - E_L,
        "resistance": 1.0,
        "noise_std": 0.0,
        "dt": 1.0,
    }

    if tag == "iafTauRefCell":
        ref = _parse_unit_value(elem.get("refract", "0ms"))
        params["refractory_period"] = int(ref)

    return ImportedCell(cell_id, "StochasticLIFNeuron", params, tag)


def _import_izhikevich_cell(elem: Any) -> ImportedCell:
    """Import <izhikevichCell> (2003 dimensionless)."""
    cell_id = elem.get("id", "unnamed")
    return ImportedCell(
        cell_id,
        "SCIzhikevichNeuron",
        {
            "a": float(elem.get("a", "0.02")),
            "b": float(elem.get("b", "0.2")),
            "c": float(elem.get("c", "-65")),
            "d": float(elem.get("d", "8")),
            "dt": 0.5,
            "noise_std": 0.0,
        },
        "izhikevichCell",
    )


def _import_izhikevich2007_cell(elem: Any) -> ImportedCell:
    """Import <izhikevich2007Cell> (biophysical units).

    Preserve the NeuroML 2 biophysical parameterisation.
    """
    cell_id = elem.get("id", "unnamed")
    C = _parse_unit_value(elem.get("C", "100pF"))
    k = _parse_unit_value(elem.get("k", "0.7"))
    vr = _parse_unit_value(elem.get("vr", "-60mV"))
    vt = _parse_unit_value(elem.get("vt", "-40mV"))
    vpeak = _parse_unit_value(elem.get("vpeak", "35mV"))
    a = _parse_unit_value(elem.get("a", "0.03"))
    b = _parse_unit_value(elem.get("b", "-2"))
    c = _parse_unit_value(elem.get("c", "-50mV"))
    d = _parse_current_pa(elem.get("d", "100pA"))
    v0 = _parse_unit_value(elem.get("v0", f"{vr}mV"))

    return ImportedCell(
        cell_id,
        "Izhikevich2007Neuron",
        {
            "C": C,
            "k": k,
            "vr": vr,
            "vt": vt,
            "vpeak": vpeak,
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "v0": v0,
            "dt": 0.1,
            "integrator": "rk4",
        },
        "izhikevich2007Cell",
    )


def _import_adex_cell(elem: Any) -> ImportedCell:
    """Import <adExIaFCell>."""
    cell_id = elem.get("id", "unnamed")
    return ImportedCell(
        cell_id,
        "AdExNeuron",
        {
            "C": _parse_unit_value(elem.get("C", "281pF")),
            "g_L": _parse_unit_value(elem.get("gL", "30nS")),
            "E_L": _parse_unit_value(elem.get("EL", "-70.6mV")),
            "V_T": _parse_unit_value(elem.get("VT", "-50.4mV")),
            "delta_T": _parse_unit_value(elem.get("delT", "2mV")),
            "tau_w": _parse_unit_value(elem.get("tauw", "144ms")),
            "a": _parse_unit_value(elem.get("a", "4nS")),
            "b": _parse_unit_value(elem.get("b", "0.0805nA")),
            "V_reset": _parse_unit_value(elem.get("reset", "-70.6mV")),
            "V_thresh": _parse_unit_value(elem.get("thresh", "-40mV")),
        },
        "adExIaFCell",
    )


_IMPORTERS = {
    "iafCell": _import_iaf_cell,
    "iafRefCell": _import_iaf_cell,
    "iafTauCell": _import_iaf_tau_cell,
    "iafTauRefCell": _import_iaf_tau_cell,
    "izhikevichCell": _import_izhikevich_cell,
    "izhikevich2007Cell": _import_izhikevich2007_cell,
    "adExIaFCell": _import_adex_cell,
}


def import_neuroml(path: str | Path) -> list[ImportedCell]:
    """Parse a NeuroML 2 XML file and return imported cell definitions.

    Parameters
    ----------
    path : str or Path
        Path to .nml or .xml file.

    Returns
    -------
    list of ImportedCell
        One per cell definition found in the file.
    """
    tree = ET.parse(path)  # nosec B314 — local file only
    root = tree.getroot()

    cells = []
    for elem in root:
        tag = _strip_ns(elem.tag)
        if tag in _IMPORTERS:
            cells.append(_IMPORTERS[tag](elem))

    return cells


def create_neuron(cell: ImportedCell) -> Any:
    """Instantiate an SC-NeuroCore neuron from an ImportedCell.

    Returns a neuron object ready for .step() calls.
    """
    if cell.cell_type == "StochasticLIFNeuron":
        from ..neurons.stochastic_lif import StochasticLIFNeuron

        safe = {k: v for k, v in cell.params.items() if not k.startswith("_")}
        return StochasticLIFNeuron(**safe)

    if cell.cell_type == "SCIzhikevichNeuron":
        from ..neurons.sc_izhikevich import SCIzhikevichNeuron

        safe = {k: v for k, v in cell.params.items() if not k.startswith("_")}
        return SCIzhikevichNeuron(**safe)

    if cell.cell_type == "Izhikevich2007Neuron":
        from ..neurons.models import Izhikevich2007Neuron

        return Izhikevich2007Neuron(**cell.params)

    if cell.cell_type == "AdExNeuron":
        from ..neurons.models.adex import AdExNeuron

        return AdExNeuron(**cell.params)

    raise ValueError(f"Unknown cell type: {cell.cell_type}")

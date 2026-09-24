# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NeuroML 2 importer

"""Import NeuroML 2 point-cell definitions into SC-NeuroCore neuron models.

Supports:
  <iafCell>, <iafRefCell>, <iafTauCell>, <iafTauRefCell> -> StochasticLIFNeuron
  <izhikevichCell> (2003 dimensionless) -> SCIzhikevichNeuron
  <izhikevich2007Cell> (biophysical) -> Izhikevich2007Neuron
  <adExIaFCell> -> AdExNeuron

NeuroML 2 spec: https://docs.neuroml.org/Userdocs/Schemas/Cells.html

Nothing is invented. Every attribute the NeuroML schema requires must be present
and carry a unit of the right physical dimension; a missing attribute, a bare
number where a unit is required, or a unit of another dimension is refused with
the cell and attribute named. A document element this importer does not model --
a network, population, projection, input, channel or morphological cell -- is
refused rather than skipped, because dropping it would silently change what the
document describes. Documentation elements (``notes``, ``annotation``,
``property``) carry no dynamics and are ignored.

Where the SC-NeuroCore model cannot take a value exactly as NeuroML states it,
the imported cell says so in ``notes``: the timestep the model runs at (NeuroML
cells carry none), voltages expressed relative to the leak reversal, the
normalised input resistance, refractory periods rounded to whole timesteps, and
attributes the target model has no place for.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET  # nosec B405
from dataclasses import dataclass
from pathlib import Path
from typing import Any

NS = "{http://www.neuroml.org/schema/neuroml2}"

# Unit -> factor to the base unit this importer uses for each NeuroML dimension:
# voltage mV, capacitance pF, conductance nS, time ms, current pA, per-time
# per_ms, conductance-per-voltage nS_per_mV.
_UNITS: dict[str, dict[str, float]] = {
    "voltage": {"V": 1e3, "mV": 1.0},
    "capacitance": {"F": 1e12, "uF": 1e6, "nF": 1e3, "pF": 1.0},
    "conductance": {"S": 1e9, "mS": 1e6, "uS": 1e3, "nS": 1.0, "pS": 1e-3},
    "time": {"s": 1e3, "ms": 1.0},
    "current": {"A": 1e12, "uA": 1e6, "nA": 1e3, "pA": 1.0},
    "pertime": {"per_s": 1e-3, "per_ms": 1.0, "Hz": 1e-3},
    "conductancePerVoltage": {"S_per_V": 1e6, "nS_per_mV": 1.0},
}
_NUMBER = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z_]*)\s*$")
_DOCUMENTATION = frozenset({"notes", "annotation", "property"})


def _strip_ns(tag: str) -> str:
    """Remove XML namespace prefix."""
    return tag.split("}")[-1] if "}" in tag else tag


def _cell_id(elem: Any) -> str:
    cell_id = elem.get("id")
    if not cell_id:
        raise ValueError(f"NeuroML <{_strip_ns(elem.tag)}> has no id; an id is required")
    return str(cell_id)


def _quantity(elem: Any, attribute: str, dimension: str) -> float:
    """Read one required attribute in the base unit of its NeuroML dimension.

    Parameters
    ----------
    elem:
        The cell element.
    attribute:
        The XML attribute name.
    dimension:
        A key of the unit table, or ``"none"`` for a dimensionless number.

    Returns
    -------
    float
        The value, finite, in the dimension's base unit.

    Raises
    ------
    ValueError
        The attribute is missing, not a number, has no unit where one is
        required, has a unit of another dimension, or is not finite.
    """
    where = f"NeuroML <{_strip_ns(elem.tag)} id={elem.get('id')!r}> attribute {attribute!r}"
    raw = elem.get(attribute)
    if raw is None:
        raise ValueError(f"{where} is missing; it is required")
    match = _NUMBER.match(raw)
    if match is None:
        raise ValueError(f"{where} value {raw!r} is not a number with a unit")
    number, unit = float(match.group(1)), match.group(2)
    if dimension == "none":
        if unit:
            raise ValueError(f"{where} is dimensionless but has unit {unit!r}")
        value = number
    else:
        units = _UNITS[dimension]
        if unit not in units:
            raise ValueError(f"{where} needs a {dimension} unit ({', '.join(units)}), got {raw!r}")
        value = number * units[unit]
    if not math.isfinite(value):
        raise ValueError(f"{where} value {raw!r} is not finite")
    return value


@dataclass
class ImportedCell:
    """Result of importing a NeuroML cell definition.

    Attributes
    ----------
    notes : tuple of str
        What the mapping to the SC-NeuroCore model assumed, approximated or
        could not carry, in words a user can check against the document.
    """

    cell_id: str
    cell_type: str
    params: dict[str, Any]
    source_tag: str
    notes: tuple[str, ...] = ()


def _refractory_steps(elem: Any, dt: float, notes: list[str]) -> int:
    refract = _quantity(elem, "refract", "time")
    steps = refract / dt
    whole = int(round(steps))
    if not math.isclose(steps, whole, rel_tol=0.0, abs_tol=1e-9):
        notes.append(
            f"refractory period {refract!r} ms realised as {whole} whole timesteps of {dt!r} ms"
        )
    return whole


def _voltage_notes(e_l: float, notes: list[str]) -> None:
    notes.append(
        f"voltages are relative to the leak reversal {e_l!r} mV, which becomes 0 in the model"
    )


def _import_iaf_cell(elem: Any) -> ImportedCell:
    """Import <iafCell> or <iafRefCell>."""
    tag = _strip_ns(elem.tag)
    cell_id = _cell_id(elem)
    dt = 1.0
    notes = [f"NeuroML cells carry no timestep; the model runs at dt={dt!r} ms"]

    C = _quantity(elem, "C", "capacitance")
    g_L = _quantity(elem, "leakConductance", "conductance")
    E_L = _quantity(elem, "leakReversal", "voltage")
    thresh = _quantity(elem, "thresh", "voltage")
    reset = _quantity(elem, "reset", "voltage")
    if g_L <= 0 or C <= 0:
        raise ValueError(f"NeuroML <{tag} id={cell_id!r}> needs positive C and leakConductance")

    # tau = C / g_L in ms, since C is in pF and g_L in nS.
    tau = C / g_L
    _voltage_notes(E_L, notes)
    notes.append(
        "input current enters through a normalised resistance of 1; the document "
        f"implies 1/leakConductance = {1e3 / g_L!r} MOhm"
    )
    params: dict[str, Any] = {
        "tau_mem": tau,
        "v_rest": 0.0,
        "v_threshold": thresh - E_L,
        "v_reset": reset - E_L,
        "resistance": 1.0,
        "noise_std": 0.0,
        "dt": dt,
    }
    if tag == "iafRefCell":
        params["refractory_period"] = _refractory_steps(elem, dt, notes)
    return ImportedCell(cell_id, "StochasticLIFNeuron", params, tag, tuple(notes))


def _import_iaf_tau_cell(elem: Any) -> ImportedCell:
    """Import <iafTauCell> or <iafTauRefCell>."""
    tag = _strip_ns(elem.tag)
    cell_id = _cell_id(elem)
    dt = 1.0
    notes = [f"NeuroML cells carry no timestep; the model runs at dt={dt!r} ms"]

    tau = _quantity(elem, "tau", "time")
    E_L = _quantity(elem, "leakReversal", "voltage")
    thresh = _quantity(elem, "thresh", "voltage")
    reset = _quantity(elem, "reset", "voltage")
    _voltage_notes(E_L, notes)
    notes.append("input current enters through a normalised resistance of 1")
    params: dict[str, Any] = {
        "tau_mem": tau,
        "v_rest": 0.0,
        "v_threshold": thresh - E_L,
        "v_reset": reset - E_L,
        "resistance": 1.0,
        "noise_std": 0.0,
        "dt": dt,
    }
    if tag == "iafTauRefCell":
        params["refractory_period"] = _refractory_steps(elem, dt, notes)
    return ImportedCell(cell_id, "StochasticLIFNeuron", params, tag, tuple(notes))


def _import_izhikevich_cell(elem: Any) -> ImportedCell:
    """Import <izhikevichCell> (2003 dimensionless)."""
    cell_id = _cell_id(elem)
    dt = 0.5
    v0 = _quantity(elem, "v0", "voltage")
    thresh = _quantity(elem, "thresh", "voltage")
    notes = (
        f"NeuroML cells carry no timestep; the model runs at dt={dt!r} ms",
        f"initial potential v0={v0!r} mV is not carried; the model starts from its own rest state",
        f"spike cut-off thresh={thresh!r} mV is not carried; the model uses its built-in 30 mV",
    )
    return ImportedCell(
        cell_id,
        "SCIzhikevichNeuron",
        {
            "a": _quantity(elem, "a", "none"),
            "b": _quantity(elem, "b", "none"),
            "c": _quantity(elem, "c", "none"),
            "d": _quantity(elem, "d", "none"),
            "dt": dt,
            "noise_std": 0.0,
        },
        "izhikevichCell",
        notes,
    )


def _import_izhikevich2007_cell(elem: Any) -> ImportedCell:
    """Import <izhikevich2007Cell> (biophysical units).

    Preserve the NeuroML 2 biophysical parameterisation.
    """
    cell_id = _cell_id(elem)
    dt = 0.1
    notes = (
        f"NeuroML cells carry no timestep; the model runs at dt={dt!r} ms",
        "integrator assumed: rk4",
    )
    return ImportedCell(
        cell_id,
        "Izhikevich2007Neuron",
        {
            "C": _quantity(elem, "C", "capacitance"),
            "k": _quantity(elem, "k", "conductancePerVoltage"),
            "vr": _quantity(elem, "vr", "voltage"),
            "vt": _quantity(elem, "vt", "voltage"),
            "vpeak": _quantity(elem, "vpeak", "voltage"),
            "a": _quantity(elem, "a", "pertime"),
            "b": _quantity(elem, "b", "conductance"),
            "c": _quantity(elem, "c", "voltage"),
            "d": _quantity(elem, "d", "current"),
            "v0": _quantity(elem, "v0", "voltage"),
            "dt": dt,
            "integrator": "rk4",
        },
        "izhikevich2007Cell",
        notes,
    )


def _import_adex_cell(elem: Any) -> ImportedCell:
    """Import <adExIaFCell> (Brette & Gerstner 2005 AdEx).

    Maps the NeuroML biophysical attributes onto the ``AdExNeuron`` constructor
    parameter names in its native, self-consistent unit system (mV, ms, pF, pA,
    nS): the leak reversal becomes ``v_rest``, the exponential threshold ``V_T``
    becomes ``v_rh``, the membrane time constant is ``tau = C / g_L`` (pF/nS = ms)
    with the capacitance kept as ``c_m`` (pF), and the spike-triggered adaptation
    ``b`` is a *current* in pA -- ``w`` and the injected current share the pA unit
    that keeps ``w / c_m`` a rate in mV/ms.
    """
    cell_id = _cell_id(elem)
    dt = 0.1
    C = _quantity(elem, "C", "capacitance")
    g_L = _quantity(elem, "gL", "conductance")
    if g_L <= 0 or C <= 0:
        raise ValueError(f"NeuroML <adExIaFCell id={cell_id!r}> needs positive C and gL")
    E_L = _quantity(elem, "EL", "voltage")
    refract = _quantity(elem, "refract", "time")
    notes = [f"NeuroML cells carry no timestep; the model runs at dt={dt!r} ms"]
    if refract != 0.0:
        notes.append(
            f"refractory period {refract!r} ms is not carried; AdExNeuron has no refractory period"
        )
    return ImportedCell(
        cell_id,
        "AdExNeuron",
        {
            "v": E_L,
            "v_rest": E_L,
            "v_reset": _quantity(elem, "reset", "voltage"),
            "v_threshold": _quantity(elem, "thresh", "voltage"),
            "v_rh": _quantity(elem, "VT", "voltage"),
            "delta_t": _quantity(elem, "delT", "voltage"),
            "tau": C / g_L,
            "tau_w": _quantity(elem, "tauw", "time"),
            "a": _quantity(elem, "a", "conductance"),
            "b": _quantity(elem, "b", "current"),
            "c_m": C,
            "dt": dt,
        },
        "adExIaFCell",
        tuple(notes),
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
    """Parse a NeuroML 2 document and return its point-cell definitions.

    Parameters
    ----------
    path : str or Path
        Path to .nml or .xml file.

    Returns
    -------
    list of ImportedCell
        One per cell definition, each with its mapping ``notes``.

    Raises
    ------
    ValueError
        The root is not a ``<neuroml>`` element, the document holds an element
        this importer does not model, or a cell attribute is missing, lacks
        its unit, has a unit of the wrong dimension or is out of range.
    """
    # The caller supplies a local NeuroML file; no remote entity is resolved.
    tree = ET.parse(path)  # nosec B314
    root = tree.getroot()
    if _strip_ns(root.tag) != "neuroml":
        raise ValueError(f"NeuroML document root must be <neuroml>, got <{_strip_ns(root.tag)}>")

    unsupported = sorted(
        {
            _strip_ns(elem.tag)
            for elem in root
            if _strip_ns(elem.tag) not in _IMPORTERS and _strip_ns(elem.tag) not in _DOCUMENTATION
        }
    )
    if unsupported:
        raise ValueError(
            "NeuroML document holds elements this importer does not model: "
            f"{', '.join(unsupported)}; it imports point-cell definitions only "
            f"({', '.join(sorted(_IMPORTERS))})"
        )
    return [
        _IMPORTERS[_strip_ns(elem.tag)](elem) for elem in root if _strip_ns(elem.tag) in _IMPORTERS
    ]


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

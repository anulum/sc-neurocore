# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio network graph to and from the Neuromorphic Intermediate Representation

"""Write a Studio network graph as a real NIR graph, and read one back.

A population becomes an NIR neuron primitive only where its model's dynamics
are that primitive's: ``SCLapicqueLIFNeuron`` (profile ``sc_lif``) is
``nir.LIF`` with ``r = resistance`` and ``v_leak = v_rest``, and
``PerfectIntegratorNeuron`` is ``nir.IF`` with ``r = 1 / c_m``. A graph with
any other model is refused, naming the population, rather than exported as a
primitive it is not.

A projection becomes ``nir.Linear`` with the connectivity the Studio runtime
realises from its rule, probability, seed and autapse decision (``weight`` is
``[target, source]``), followed by ``nir.Delay``: the runtime delivers a spike
``delay + 1`` steps after it was emitted, so the delay carries that latency.
NIR has no stimulus primitive, so a driven population receives an
``nir.Input``; the drive itself is stated in the export notes and kept in the
node metadata. Times are in milliseconds, the Studio graph's unit; NIR records
no unit.

Every node also carries the Studio fields it came from under
:data:`STUDIO_METADATA_KEY`, so a file Studio wrote reads back as the same
network. Reading such a file rebuilds the graph from that metadata and then
checks that re-exporting it reproduces every tensor in the file; an edited
tensor is refused, not silently overridden. A file from another tool is read
only where the Studio graph can hold it: populations of uniform ``nir.LIF`` or
``nir.IF`` neurons joined by all-to-all ``nir.Linear`` weights of one value,
with an optional ``nir.Delay`` of whole steps.
"""

from __future__ import annotations

import base64
import binascii
import json
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sc_neurocore.studio.network_execution import PROJECTION_LATENCY_STEPS, connectivity_arrays
from sc_neurocore.studio.network_graph_spec import (
    DEFAULT_DT_MS,
    GraphRejected,
    GraphSpec,
    PopulationSpec,
    ProjectionSpec,
    resolve_graph,
)

STUDIO_NIR_EXPORT_SCHEMA_VERSION = "sc-neurocore.studio.nir-export.v1"
STUDIO_METADATA_KEY = "sc_neurocore.studio"
NIR_EXPORT_FILENAME = "network.nir"
_LIF_MODEL = "SCLapicqueLIFNeuron"
_IF_MODEL = "PerfectIntegratorNeuron"
# PerfectIntegratorNeuron's spike comparison per profile (perfect_integrator.py).
_IF_COMPARATORS = {"sc_inclusive": ">=", "naud_gerstner_2012": ">"}
_TIME_UNIT_NOTE = (
    "times (tau, delay) are in milliseconds, the Studio graph's unit; NIR records no unit, "
    "so a consumer that assumes seconds must scale them"
)


class NIRMappingRefused(ValueError):
    """A graph or file holds something the other side cannot represent."""


@dataclass(frozen=True, slots=True)
class NIRExport:
    """One written NIR file and what the writing could not carry exactly."""

    content: bytes
    notes: tuple[str, ...]
    nir_version: str

    def to_public_dict(self) -> dict[str, object]:
        """Return the JSON projection the Studio API serves."""
        return {
            "schema_version": STUDIO_NIR_EXPORT_SCHEMA_VERSION,
            "filename": NIR_EXPORT_FILENAME,
            "media_type": "application/x-hdf5",
            "nir_version": self.nir_version,
            "content_base64": base64.b64encode(self.content).decode("ascii"),
            "notes": list(self.notes),
        }


def graph_to_nir_file(graph: object) -> NIRExport:
    """Export a Studio network graph as a real NIR (HDF5) file.

    Raises
    ------
    GraphRejected
        When the graph does not validate.
    NIRMappingRefused
        When a population's model is not an NIR primitive.
    """
    nir = _nir()
    graph_mapping = graph if isinstance(graph, Mapping) else {}
    spec = resolve_graph(graph)
    nir_graph, notes = _to_nir_graph(nir, spec, graph_mapping)
    with tempfile.TemporaryDirectory(prefix="sc_studio_nir_") as directory:
        path = Path(directory) / NIR_EXPORT_FILENAME
        nir.write(path, nir_graph)
        content = path.read_bytes()
    return NIRExport(content=content, notes=tuple(notes), nir_version=str(nir.version))


def nir_file_to_graph(content_base64: object) -> dict[str, Any]:
    """Read an NIR file (base64 of its bytes) into a Studio network graph.

    Returns
    -------
    dict
        ``graph`` (the Studio graph), ``origin`` (``studio`` or ``foreign``)
        and ``notes`` (what the reading assumed).

    Raises
    ------
    NIRMappingRefused
        When the file is not NIR, holds a node the graph cannot represent, or
        its tensors do not match the Studio network its metadata describes.
    """
    if not isinstance(content_base64, str) or not content_base64:
        raise NIRMappingRefused("an NIR import needs the file's bytes as base64 text")
    try:
        content = base64.b64decode(content_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise NIRMappingRefused(f"the NIR import is not valid base64: {exc}") from exc
    nir = _nir()
    with tempfile.TemporaryDirectory(prefix="sc_studio_nir_") as directory:
        path = Path(directory) / NIR_EXPORT_FILENAME
        path.write_bytes(content)
        try:
            nir_graph = nir.read(path)
        except (OSError, KeyError, ValueError, TypeError, AttributeError) as exc:
            raise NIRMappingRefused(f"the file is not a readable NIR graph: {exc}") from exc
    studio = _studio_metadata(nir_graph.metadata)
    if studio is not None:
        return _studio_graph(nir, nir_graph, studio)
    return _foreign_graph(nir, nir_graph)


def _nir() -> Any:
    """Import the NIR reference package, which the ``nir`` extra installs."""
    try:
        import nir
    except ImportError as exc:  # the optional extra is absent
        raise NIRMappingRefused(
            "NIR export and import need the nir package: install sc-neurocore[nir]"
        ) from exc
    return nir


def _to_nir_graph(nir: Any, spec: GraphSpec, graph: Mapping[str, Any]) -> tuple[Any, list[str]]:
    """Build the NIR graph of ``spec`` and the notes on what it does not carry."""
    raw_populations = {
        str(item.get("id")): item
        for item in graph.get("populations", [])
        if isinstance(item, Mapping)
    }
    nodes: dict[str, Any] = {}
    edges: list[tuple[str, str]] = []
    notes = [_TIME_UNIT_NOTE]
    counts = {population.id: population.count for population in spec.populations}
    for population in spec.populations:
        node, population_notes = _neuron_primitive(nir, population)
        node.metadata[STUDIO_METADATA_KEY] = json.dumps(
            _population_metadata(population, raw_populations.get(population.id, {})),
            sort_keys=True,
        )
        nodes[population.id] = node
        notes.extend(population_notes)
        output = f"output_{population.id}"
        nodes[output] = nir.Output(output_type={"output": np.array([population.count])})
        edges.append((population.id, output))
        if population.drive.kind != "none":
            source = f"input_{population.id}"
            nodes[source] = nir.Input(input_type={"input": np.array([population.count])})
            edges.append((source, population.id))
            notes.append(_drive_note(population))
    for projection in spec.projections:
        weight_name = f"{projection.id}_weight"
        delay_name = f"{projection.id}_delay"
        weight = _dense_weight(projection, counts[projection.source], counts[projection.target])
        nodes[weight_name] = nir.Linear(weight=weight)
        nodes[weight_name].metadata[STUDIO_METADATA_KEY] = json.dumps(
            _projection_metadata(projection), sort_keys=True
        )
        latency_ms = (projection.delay_steps + PROJECTION_LATENCY_STEPS) * spec.dt
        nodes[delay_name] = nir.Delay(delay=np.full(counts[projection.target], latency_ms))
        edges.extend(
            [
                (projection.source, weight_name),
                (weight_name, delay_name),
                (delay_name, projection.target),
            ]
        )
    notes.append(
        f"each projection's nir.Delay includes the Studio runtime's {PROJECTION_LATENCY_STEPS}-step "
        f"spike propagation latency at dt = {spec.dt!r} ms"
    )
    metadata = {
        STUDIO_METADATA_KEY: json.dumps(
            {
                "schema_version": STUDIO_NIR_EXPORT_SCHEMA_VERSION,
                "dt": spec.dt,
                "duration": spec.duration_ms,
                "seed": spec.seed,
                "time_unit": "ms",
            },
            sort_keys=True,
        )
    }
    return nir.NIRGraph(nodes=nodes, edges=edges, metadata=metadata), notes


def _neuron_primitive(nir: Any, population: PopulationSpec) -> tuple[Any, list[str]]:
    """Return the NIR primitive of one population and notes on its mapping."""
    neuron = population.inputs.instantiate()
    count = population.count
    notes: list[str] = []
    if population.model == _LIF_MODEL and neuron.profile == "sc_lif":
        node = nir.LIF(
            tau=np.full(count, float(neuron.tau)),
            r=np.full(count, float(neuron.resistance)),
            v_leak=np.full(count, float(neuron.v_rest)),
            v_threshold=np.full(count, float(neuron.v_threshold)),
            v_reset=np.full(count, float(neuron.v_reset)),
        )
        notes.append(
            f"population {population.id}: the model fires at v >= v_threshold, NIR LIF at "
            "v > v_threshold; Studio integrates it exactly per step"
        )
        resting = float(neuron.v_rest)
    elif population.model == _IF_MODEL:
        node = nir.IF(
            r=np.full(count, 1.0 / float(neuron.c_m)),
            v_threshold=np.full(count, float(neuron.v_threshold)),
            v_reset=np.full(count, float(neuron.v_reset)),
        )
        notes.append(
            f"population {population.id}: profile {neuron.profile} fires at "
            f"v {_IF_COMPARATORS[neuron.profile]} v_threshold, NIR IF at v > v_threshold"
        )
        resting = 0.0
    else:
        raise NIRMappingRefused(
            f"population {population.id}: model {population.model}"
            + (f" (profile {neuron.profile})" if hasattr(neuron, "profile") else "")
            + " has no NIR primitive with the same dynamics; NIR export maps "
            f"{_LIF_MODEL} (profile sc_lif) to nir.LIF and {_IF_MODEL} to nir.IF"
        )
    if float(neuron.v) != resting:
        notes.append(
            f"population {population.id}: initial membrane {float(neuron.v)!r} is not carried; "
            "NIR primitives have no initial state"
        )
    return node, notes


def _drive_note(population: PopulationSpec) -> str:
    drive = population.drive
    if drive.kind == "constant":
        stated = f"constant current {drive.current!r}"
    else:
        stated = (
            f"Poisson input at {drive.rate_hz!r} Hz, weight {drive.weight!r}, seed {drive.seed!r}"
        )
    return (
        f"population {population.id}: its drive ({stated}) is not an NIR primitive; the "
        f"file routes it through input node input_{population.id} and records it in metadata"
    )


def _dense_weight(projection: ProjectionSpec, n_source: int, n_target: int) -> np.ndarray[Any, Any]:
    """Return the realised connectivity as a dense ``[target, source]`` matrix."""
    indptr, indices, data, _removed = connectivity_arrays(projection, n_source, n_target)
    weight = np.zeros((n_target, n_source))
    rows = np.repeat(np.arange(n_source), np.diff(indptr))
    weight[indices, rows] = data
    return weight


def _population_metadata(population: PopulationSpec, raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": "population",
        "index": population.index,
        "label": population.label,
        "model": population.model,
        "count": population.count,
        "neuron_type": population.neuron_type,
        "params": dict(raw.get("params", {})) if isinstance(raw.get("params"), Mapping) else {},
        "drive": dict(raw["drive"]) if isinstance(raw.get("drive"), Mapping) else {"kind": "none"},
        "position": dict(raw["position"]) if isinstance(raw.get("position"), Mapping) else None,
    }


def _projection_metadata(projection: ProjectionSpec) -> dict[str, Any]:
    return {
        "kind": "projection",
        "index": projection.index,
        "id": projection.id,
        "source": projection.source,
        "target": projection.target,
        "weight": projection.weight,
        "rule": projection.rule,
        "probability": projection.probability,
        "seed": projection.seed,
        "autapses": projection.autapses,
        "delay": projection.delay_ms,
    }


def _studio_metadata(metadata: Mapping[str, Any]) -> dict[str, Any] | None:
    raw = metadata.get(STUDIO_METADATA_KEY)
    if raw is None:
        return None
    decoded = _decode_metadata(raw, "the graph")
    if decoded.get("schema_version") != STUDIO_NIR_EXPORT_SCHEMA_VERSION:
        raise NIRMappingRefused(
            f"the file names Studio metadata of an unknown version: {decoded!r}"
        )
    return decoded


def _decode_metadata(raw: object, where: str) -> dict[str, Any]:
    """Return one Studio metadata object, refusing text that is not a JSON object."""
    try:
        decoded = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError as exc:
        raise NIRMappingRefused(f"{where}: its Studio metadata is not JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise NIRMappingRefused(f"{where}: its Studio metadata is not an object: {decoded!r}")
    return decoded


def _studio_graph(nir: Any, nir_graph: Any, studio: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild a Studio-written graph and refuse it if its tensors were edited."""
    # HDF5 returns nodes by name; the graph's order is kept in each entry's index,
    # because seeds the graph does not state are derived from that order.
    populations: list[tuple[int, dict[str, Any]]] = []
    projections: list[tuple[int, dict[str, Any]]] = []
    try:
        for name, node in nir_graph.nodes.items():
            raw = node.metadata.get(STUDIO_METADATA_KEY)
            if raw is None:
                continue
            entry = _decode_metadata(raw, f"node {name}")
            if entry.get("kind") == "population":
                populations.append((entry["index"], _studio_population(name, entry)))
            elif entry.get("kind") == "projection":
                projections.append((entry["index"], _studio_projection(entry)))
            else:
                raise NIRMappingRefused(
                    f"node {name}: its Studio metadata names an unknown kind {entry.get('kind')!r}"
                )
        graph = {
            "populations": [item for _index, item in sorted(populations, key=_first)],
            "projections": [item for _index, item in sorted(projections, key=_first)],
            "dt": studio["dt"],
            "duration": studio["duration"],
            "seed": studio["seed"],
        }
    except KeyError as exc:
        raise NIRMappingRefused(f"the file's Studio metadata lacks the field {exc}") from exc
    try:
        spec = resolve_graph(graph)
    except GraphRejected as exc:
        raise NIRMappingRefused(
            f"the network the file's Studio metadata describes does not validate: {exc}"
        ) from exc
    rebuilt, _notes = _to_nir_graph(nir, spec, graph)
    if set(rebuilt.nodes) != set(nir_graph.nodes) or sorted(rebuilt.edges) != sorted(
        nir_graph.edges
    ):
        raise NIRMappingRefused(
            "the file's nodes or edges differ from the Studio network its metadata describes"
        )
    for name, node in rebuilt.nodes.items():
        if not _same_tensors(node, nir_graph.nodes[name]):
            raise NIRMappingRefused(
                f"node {name} does not match the Studio network its metadata describes; "
                "the file was changed after Studio wrote it"
            )
    return {"graph": graph, "origin": "studio", "notes": []}


def _first(item: tuple[int, dict[str, Any]]) -> int:
    return item[0]


def _studio_population(name: str, entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": name,
        "type": "population",
        "label": entry["label"],
        "model": entry["model"],
        "count": entry["count"],
        "neuron_type": entry["neuron_type"],
        "params": entry["params"],
        "drive": entry["drive"],
        "position": entry["position"] or {"x": 200 * entry["index"], "y": 0},
    }


def _studio_projection(entry: Mapping[str, Any]) -> dict[str, Any]:
    projection = {
        key: entry[key]
        for key in ("id", "source", "target", "weight", "rule", "seed", "autapses", "delay")
    }
    if entry["rule"] == "random":
        projection["probability"] = entry["probability"]
    return projection


def _same_tensors(expected: Any, actual: Any) -> bool:
    """Compare two NIR nodes' type and every array field, ignoring metadata."""
    if type(expected) is not type(actual):
        return False
    for field, value in vars(expected).items():
        if field == "metadata" or not isinstance(value, np.ndarray):
            continue
        if not np.array_equal(getattr(actual, field), value):
            return False
    return True


def _foreign_graph(nir: Any, nir_graph: Any) -> dict[str, Any]:
    """Read a file another tool wrote, where the Studio graph can hold it."""
    populations: dict[str, dict[str, Any]] = {}
    notes = [_TIME_UNIT_NOTE]
    for name, node in nir_graph.nodes.items():
        if isinstance(node, (nir.Input, nir.Output, nir.Linear, nir.Delay)):
            continue
        if isinstance(node, nir.LIF):
            params = _uniform(name, node, ("tau", "r", "v_leak", "v_threshold", "v_reset"))
            populations[name] = _foreign_population(
                name,
                _LIF_MODEL,
                node.tau,
                {
                    "tau": params["tau"],
                    "resistance": params["r"],
                    "v_rest": params["v_leak"],
                    "v_threshold": params["v_threshold"],
                    "v_reset": params["v_reset"],
                },
            )
            notes.append(
                f"{name}: NIR LIF fires at v > v_threshold, {_LIF_MODEL} at v >= v_threshold"
            )
        elif isinstance(node, nir.IF):
            params = _uniform(name, node, ("r", "v_threshold", "v_reset"))
            populations[name] = _foreign_population(
                name,
                _IF_MODEL,
                node.r,
                {
                    "c_m": 1.0 / params["r"],
                    "v_threshold": params["v_threshold"],
                    "v_reset": params["v_reset"],
                },
            )
            notes.append(
                f"{name}: NIR IF fires at v > v_threshold, {_IF_MODEL} at v >= v_threshold"
            )
        else:
            raise NIRMappingRefused(
                f"{name}: {type(node).__name__} has no Studio population or projection; "
                "the Studio graph reads nir.LIF and nir.IF neurons joined by nir.Linear"
            )
    projections = _foreign_projections(nir, nir_graph, populations, notes)
    _foreign_neuron_types(populations, projections, notes)
    graph = {
        "populations": list(populations.values()),
        "projections": projections,
        "dt": DEFAULT_DT_MS,
    }
    try:
        resolve_graph(graph)
    except GraphRejected as exc:
        raise NIRMappingRefused(f"the imported network does not validate: {exc}") from exc
    return {"graph": graph, "origin": "foreign", "notes": notes}


def _foreign_neuron_types(
    populations: Mapping[str, dict[str, Any]],
    projections: list[dict[str, Any]],
    notes: list[str],
) -> None:
    """Type each population by the sign of its outgoing weights (Dale's principle)."""
    signs: dict[str, set[bool]] = {}
    for projection in projections:
        signs.setdefault(projection["source"], set()).add(projection["weight"] > 0)
    for name, positive in signs.items():
        if len(positive) > 1:
            raise NIRMappingRefused(
                f"{name}: its outgoing weights have both signs; a Studio population is "
                "either excitatory or inhibitory"
            )
        if positive == {False}:
            populations[name]["neuron_type"] = "inhibitory"
            notes.append(f"{name}: read as inhibitory, its outgoing weights are negative")


def _uniform(name: str, node: Any, fields: tuple[str, ...]) -> dict[str, float]:
    """Return each field's single value, refusing per-neuron values."""
    values: dict[str, float] = {}
    for field in fields:
        # NIR itself fills an omitted v_reset with zeros, so every field is an array.
        array = np.atleast_1d(np.asarray(getattr(node, field), dtype=np.float64))
        if array.size == 0 or not np.all(array == array.flat[0]):
            raise NIRMappingRefused(
                f"{name}: {field} differs between neurons; a Studio population has one value"
            )
        values[field] = float(array.flat[0])
    return values


def _foreign_population(
    name: str, model: str, sized: Any, params: dict[str, Any]
) -> dict[str, Any]:
    return {
        "id": name,
        "type": "population",
        "label": name,
        "model": model,
        "count": int(np.atleast_1d(np.asarray(sized)).size),
        "neuron_type": "excitatory",
        "params": params,
        "drive": {"kind": "none"},
        "position": {"x": 0, "y": 0},
    }


def _foreign_projections(
    nir: Any, nir_graph: Any, populations: Mapping[str, Any], notes: list[str]
) -> list[dict[str, Any]]:
    """Read population -> Linear -> (Delay ->) population chains as projections."""
    successors: dict[str, list[str]] = {}
    for source, target in nir_graph.edges:
        successors.setdefault(source, []).append(target)
    projections: list[dict[str, Any]] = []
    for source in populations:
        for weight_name in successors.get(source, []):
            node = nir_graph.nodes[weight_name]
            if isinstance(node, nir.Output):
                continue
            if not isinstance(node, nir.Linear):
                raise NIRMappingRefused(
                    f"{source} -> {weight_name}: a population may feed only nir.Linear or nir.Output"
                )
            for after in successors.get(weight_name, []):
                latency_ms = PROJECTION_LATENCY_STEPS * DEFAULT_DT_MS
                if isinstance(nir_graph.nodes[after], nir.Delay):
                    total = _uniform(after, nir_graph.nodes[after], ("delay",))["delay"]
                    if total < latency_ms:
                        raise NIRMappingRefused(
                            f"{after}: a delay of {total!r} ms is shorter than the Studio "
                            f"runtime's {latency_ms!r} ms propagation latency"
                        )
                    delay_ms = total - latency_ms
                    targets = successors.get(after, [])
                    target = targets[0] if len(targets) == 1 else ""
                else:
                    delay_ms, target = 0.0, after
                    notes.append(
                        f"{weight_name}: the Studio runtime adds a {PROJECTION_LATENCY_STEPS}-step "
                        "propagation latency the file's instantaneous edge does not have"
                    )
                if target not in populations:
                    raise NIRMappingRefused(
                        f"{weight_name}: its output reaches {target}, which is not a population"
                    )
                weight = np.asarray(node.weight)
                if weight.size == 0 or not np.all(weight == weight.flat[0]) or weight.flat[0] == 0:
                    raise NIRMappingRefused(
                        f"{weight_name}: the Studio graph holds only all-to-all projections of one "
                        "weight or seeded random ones it writes itself; this matrix is neither"
                    )
                projections.append(
                    {
                        "id": weight_name,
                        "source": source,
                        "target": target,
                        "weight": float(weight.flat[0]),
                        "delay": delay_ms,
                        "rule": "all_to_all",
                    }
                )
    return projections


__all__ = [
    "NIR_EXPORT_FILENAME",
    "STUDIO_METADATA_KEY",
    "STUDIO_NIR_EXPORT_SCHEMA_VERSION",
    "NIRExport",
    "NIRMappingRefused",
    "graph_to_nir_file",
    "nir_file_to_graph",
]

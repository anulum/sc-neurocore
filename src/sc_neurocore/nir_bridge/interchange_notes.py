# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — What a NIR import or export assumed, approximated or left out

"""Say what crossing the NIR boundary assumed, approximated or did not carry.

NIR describes a graph and its parameters, not everything an executable network
needs, and this bridge realises a few constructs in its own way. Each such
point is recorded as a note rather than decided silently:

``assumed``
    A value NIR does not store and the bridge supplied: the timestep, the reset
    rule of spiking nodes, the one-step delay that realises a recurrent edge.
``approximated``
    A value NIR does store that the bridge could not realise exactly: a delay
    that is not a whole number of timesteps, a port beyond the first.
``not-carried``
    Something present on one side that the other side does not keep: graph
    metadata on import; the timestep, a subtract reset, node state or pooling
    input shapes on export.

Notes never replace a refusal: a node type the bridge cannot realise is still
rejected.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import nir
import numpy as np

NoteKind = Literal["assumed", "approximated", "not-carried"]

_SPIKING = (nir.LIF, nir.IF, nir.CubaLIF)
_SPIKING_NAMES = frozenset({"SCLIFNode", "SCIFNode", "SCCubaLIFNode"})
_STATEFUL_NAMES = frozenset(
    {
        "SCLIFNode",
        "SCIFNode",
        "SCLINode",
        "SCIntegratorNode",
        "SCCubaLIFNode",
        "SCCubaLINode",
        "SCDelayNode",
    }
)
_POOLING_NAMES = frozenset({"SCSumPool2dNode", "SCAvgPool2dNode"})


@dataclass(frozen=True, slots=True)
class InterchangeNote:
    """One point where crossing the NIR boundary was not exact.

    Parameters
    ----------
    kind:
        ``assumed``, ``approximated`` or ``not-carried``.
    subject:
        The graph (``""``) or the dotted path of the node or edge concerned.
    detail:
        What happened, in words a user can check against the source graph.
    """

    kind: NoteKind
    subject: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        """Return the note as a JSON-compatible mapping."""
        return {"kind": self.kind, "subject": self.subject, "detail": self.detail}


def read_nir_file_version(path: str | Path) -> str | None:
    """Return the NIR version a ``.nir`` file records, if it records one.

    Parameters
    ----------
    path:
        The ``.nir`` (HDF5) file.

    Returns
    -------
    str or None
        The version string, or ``None`` when the file stores no version.
    """
    import h5py

    with h5py.File(str(path), "r") as handle:
        if "version" not in handle:
            return None
        raw = handle["version"][()]
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


def _path(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _contains_spiking(graph: nir.NIRGraph) -> bool:
    for node in graph.nodes.values():
        if isinstance(node, _SPIKING):
            return True
        if isinstance(node, nir.NIRGraph) and _contains_spiking(node):
            return True
    return False


def import_notes(
    graph: nir.NIRGraph, network: Any, *, dt: float, reset_mode: str
) -> list[InterchangeNote]:
    """Record what importing ``graph`` as ``network`` assumed or approximated.

    Parameters
    ----------
    graph:
        The NIR graph that was imported.
    network:
        The parsed ``SCNetwork``, before any execution.
    dt:
        The timestep the importer used.
    reset_mode:
        The reset rule the importer gave spiking nodes.

    Returns
    -------
    list of InterchangeNote
        Graph-level notes first, then per-node and per-edge notes in graph order.
    """
    notes = [
        InterchangeNote(
            "assumed",
            "",
            f"NIR stores no timestep; dynamics use dt={float(dt)!r} supplied to the importer",
        )
    ]
    if _contains_spiking(graph):
        notes.append(
            InterchangeNote(
                "assumed",
                "",
                f"NIR stores no reset rule; spiking nodes use {reset_mode!r} "
                "(the NIR convention is 'reset' to v_reset)",
            )
        )
    notes.extend(_graph_notes(graph, network, dt=dt, prefix=""))
    return notes


def _graph_notes(
    graph: nir.NIRGraph, network: Any, *, dt: float, prefix: str
) -> list[InterchangeNote]:
    notes: list[InterchangeNote] = []
    metadata = getattr(graph, "metadata", None) or {}
    if metadata:
        keys = ", ".join(sorted(str(key) for key in metadata))
        notes.append(
            InterchangeNote("not-carried", prefix, f"graph metadata is not imported: {keys}")
        )
    for name, node in graph.nodes.items():
        subject = _path(prefix, name)
        if isinstance(node, nir.NIRGraph):
            sub_network = network.nodes[name].network
            if len(sub_network.input_nodes) != 1 or len(sub_network.output_nodes) != 1:
                notes.append(
                    InterchangeNote(
                        "approximated",
                        subject,
                        "multi-port subgraph used as one node: only its first input and "
                        "first output port are connected in the enclosing graph",
                    )
                )
            notes.extend(_graph_notes(node, sub_network, dt=dt, prefix=subject))
            continue
        if isinstance(node, nir.Input) and len(node.input_type) > 1:
            notes.append(
                InterchangeNote(
                    "approximated", subject, "only the first input port's shape is used"
                )
            )
        if isinstance(node, nir.Output) and len(node.output_type) > 1:
            notes.append(
                InterchangeNote(
                    "approximated", subject, "only the first output port's shape is used"
                )
            )
        if isinstance(node, nir.Delay):
            requested = np.atleast_1d(np.asarray(node.delay, dtype=np.float64)).flatten() / float(
                dt
            )
            realised = np.round(requested)
            if not np.allclose(requested, realised, rtol=0.0, atol=1e-9):
                notes.append(
                    InterchangeNote(
                        "approximated",
                        subject,
                        f"delay of {requested.tolist()} timesteps realised as "
                        f"{realised.astype(int).tolist()} whole timesteps",
                    )
                )
    for source, destination in network._find_back_edges():
        notes.append(
            InterchangeNote(
                "assumed",
                _path(prefix, f"{source}->{destination}"),
                "recurrent edge realised with a one-timestep delay",
            )
        )
    return notes


def export_notes(network: Any) -> list[InterchangeNote]:
    """Record what exporting ``network`` to NIR does not carry.

    Parameters
    ----------
    network:
        The ``SCNetwork`` about to be exported.

    Returns
    -------
    list of InterchangeNote
        Graph-level notes first, then per-node notes.
    """
    classes = set(_node_class_names(network))
    notes = [
        InterchangeNote(
            "not-carried",
            "",
            f"NIR stores no timestep; dt={float(network.dt)!r} must be supplied again on import",
        )
    ]
    if network.reset_mode == "subtract" and classes & _SPIKING_NAMES:
        notes.append(
            InterchangeNote(
                "not-carried",
                "",
                "NIR stores no reset rule; the subtract reset is lost and importers "
                "apply the NIR convention of reset to v_reset",
            )
        )
    if classes & _STATEFUL_NAMES:
        notes.append(
            InterchangeNote(
                "not-carried",
                "",
                "NIR stores no node state; membrane, current and delay-buffer state is "
                "not written and an importer starts from the initial state",
            )
        )
    notes.extend(_pooling_notes(network, prefix=""))
    return notes


def _node_class_names(network: Any) -> Iterable[str]:
    for node in network.nodes.values():
        yield type(node).__name__
        inner = getattr(node, "network", None)
        if inner is not None:
            yield from _node_class_names(inner)


def _pooling_notes(network: Any, *, prefix: str) -> list[InterchangeNote]:
    notes: list[InterchangeNote] = []
    for name, node in network.nodes.items():
        subject = _path(prefix, name)
        inner = getattr(node, "network", None)
        if inner is not None:
            notes.extend(_pooling_notes(inner, prefix=subject))
        elif type(node).__name__ in _POOLING_NAMES:
            notes.append(
                InterchangeNote(
                    "not-carried", subject, "pooling input shape (input_type) is not written"
                )
            )
    return notes


__all__ = [
    "InterchangeNote",
    "NoteKind",
    "export_notes",
    "import_notes",
    "read_nir_file_version",
]

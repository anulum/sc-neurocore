# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quantum Studio visualisation hook

"""Telemetry hook for the quantum cognition layer.

Provides structured metadata and real-time data for integration with
SNN Visual Studio and SCPN Studio frontends.  All outputs are plain
Python dicts / lists suitable for JSON serialisation.
"""

import json
import time
from dataclasses import dataclass
from typing import Any

from .bridge_adapter import FisherPosnerQuantumBridge
from .spin_pool import SpinPoolMPS


@dataclass(frozen=True)
class QuantumCognitionLayerMetadata:
    """Structured metadata for the quantum cognition visualisation layer."""

    layer_name: str = "Quantum Cognition (Fisher-Posner)"
    status: str = "stable"
    avg_entanglement: float = 0.0
    n_sites: int = 0
    color: str = "#00f2ff"
    node_style: str = "glow"

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return {
            "layer_name": self.layer_name,
            "status": self.status,
            "metrics": {
                "avg_entanglement": self.avg_entanglement,
                "n_sites": self.n_sites,
            },
            "visual_config": {
                "color": self.color,
                "node_style": self.node_style,
            },
        }


class QuantumStudioHook:
    """Telemetry endpoint for quantum cognition layer visualisation.

    Provides structured metadata and streaming data for SNN Visual Studio
    and SCPN Studio frontends.

    Parameters
    ----------
    spin_pool : SpinPoolMPS
        The spin pool to observe.
    bridge : FisherPosnerQuantumBridge
        The quantum bridge to observe.
    """

    def __init__(
        self,
        spin_pool: SpinPoolMPS,
        bridge: FisherPosnerQuantumBridge,
    ) -> None:
        self.spin_pool = spin_pool
        self.bridge = bridge

    def get_layer_metadata(self) -> QuantumCognitionLayerMetadata:
        """Return layer metadata for the Studio layer panel."""
        status = self.spin_pool.get_status()
        return QuantumCognitionLayerMetadata(
            status=status["coherence_status"],
            avg_entanglement=status["avg_entanglement"],
            n_sites=status["n_sites"],
        )

    def get_layer_metadata_dict(self) -> dict[str, Any]:
        """Return layer metadata as a plain dict (JSON-serialisable)."""
        return self.get_layer_metadata().to_dict()

    def get_realtime_data(self) -> dict[str, Any]:
        """Return streaming data for live entanglement graph.

        Returns
        -------
        dict
            Contains ``entanglement_map`` (list[float]) and
            ``atp_efficiencies`` (list[float]) for all sites.
        """
        pool = self.spin_pool
        return {
            "entanglement_map": pool.entanglement_map.tolist(),
            "atp_efficiencies": [pool.get_local_atp_efficiency(i) for i in range(pool.n_sites)],
            "bridge_backend": self.bridge.backend,
            "bridge_n_qubits": self.bridge.n_qubits,
        }

    def get_entanglement_snapshot(self) -> dict[str, Any]:
        """Return a timestamped snapshot of entanglement and ATP state.

        Suitable for logging, archiving, or streaming to external
        dashboard frontends via the Studio JSON API.

        Returns
        -------
        dict
            Timestamped snapshot with entanglement map, ATP levels,
            measurement count, and status summary.
        """
        pool = self.spin_pool
        status = pool.get_status()
        return {
            "timestamp": time.time(),
            "n_sites": pool.n_sites,
            "entanglement_map": pool.entanglement_map.tolist(),
            "atp_efficiencies": [pool.get_local_atp_efficiency(i) for i in range(pool.n_sites)],
            "avg_entanglement": status["avg_entanglement"],
            "max_entanglement": status["max_entanglement"],
            "min_entanglement": status["min_entanglement"],
            "measurement_count": status["measurement_count"],
            "coherence_status": status["coherence_status"],
            "bridge_backend": self.bridge.backend,
        }

    def to_json_event(self, event_type: str = "snapshot") -> str:
        """Produce a JSON event string for external frontend streaming.

        Returns a newline-delimited JSON (NDJSON) line suitable for
        Server-Sent Events or NDJSON log appending.

        Parameters
        ----------
        event_type : str
            Event type label (default ``"snapshot"``).

        Returns
        -------
        str
            Single-line JSON string with ``event``, ``data``, and
            ``timestamp`` fields.
        """
        snapshot = self.get_entanglement_snapshot()
        event = {
            "event": event_type,
            "data": snapshot,
            "timestamp": snapshot["timestamp"],
        }
        return json.dumps(event, separators=(",", ":"))

    def __repr__(self) -> str:
        return f"QuantumStudioHook(spin_pool={self.spin_pool!r}, bridge={self.bridge!r})"


__all__ = ["QuantumStudioHook", "QuantumCognitionLayerMetadata"]

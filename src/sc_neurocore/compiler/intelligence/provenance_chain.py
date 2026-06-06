# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Provenance chain

"""Cryptographic audit trail generation for neuron compilation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


@dataclass
class ProvenanceRecord:
    """Cryptographic audit trail entry.

    Attributes
    ----------
    stage : str
        Pipeline stage name.
    input_hash : str
        SHA-256 of input artefact.
    output_hash : str
        SHA-256 of output artefact.
    timestamp : str
        ISO 8601 timestamp.
    parameters : dict
        Compilation parameters used.
    """

    stage: str
    input_hash: str
    output_hash: str
    timestamp: str
    parameters: dict


def generate_provenance_chain(
    module_name: str,
    equations: dict[str, str],
    verilog_source: str = "",
    *,
    target: str = "artix7",
    data_width: int = 16,
    fraction: int = 8,
) -> list[ProvenanceRecord]:
    """Generate a cryptographic provenance chain for compilation."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()

    eq_str = json.dumps(equations, sort_keys=True)
    eq_hash = hashlib.sha256(eq_str.encode()).hexdigest()

    params = {
        "module_name": module_name,
        "target": target,
        "data_width": data_width,
        "fraction": fraction,
    }
    params_str = json.dumps(params, sort_keys=True)
    params_hash = hashlib.sha256(params_str.encode()).hexdigest()

    v_hash = hashlib.sha256(verilog_source.encode()).hexdigest()

    chain = [
        ProvenanceRecord(
            stage="source_equations",
            input_hash="genesis",
            output_hash=eq_hash,
            timestamp=now,
            parameters={"equation_count": len(equations)},
        ),
        ProvenanceRecord(
            stage="compilation_config",
            input_hash=eq_hash,
            output_hash=params_hash,
            timestamp=now,
            parameters=params,
        ),
        ProvenanceRecord(
            stage="verilog_generation",
            input_hash=params_hash,
            output_hash=v_hash,
            timestamp=now,
            parameters={"verilog_lines": verilog_source.count("\n") + 1},
        ),
    ]

    return chain


def format_provenance_json(chain: list[ProvenanceRecord]) -> str:
    """Format provenance chain as JSON manifest."""

    data = {
        "sc_neurocore_provenance": {
            "version": "1.0",
            "chain": [
                {
                    "stage": r.stage,
                    "input_hash": r.input_hash,
                    "output_hash": r.output_hash,
                    "timestamp": r.timestamp,
                    "parameters": r.parameters,
                }
                for r in chain
            ],
        }
    }
    return json.dumps(data, indent=2)

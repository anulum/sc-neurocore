#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — isolated NIR graph lowering benchmark probe

"""Emit one cold-process NIR-to-hardware-graph fidelity and timing sample."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import resource
import statistics
import sys
import time
from typing import Any

import nir
import numpy as np


def _nir_graph() -> Any:
    """Build a deterministic two-population NIR graph."""
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([4])}),
            "affine_hidden": nir.Affine(
                weight=np.array(
                    [
                        [0.25, -0.5, 0.125, 0.75],
                        [-0.125, 0.375, 0.5, -0.25],
                        [0.625, 0.125, -0.375, 0.25],
                        [-0.5, 0.25, 0.75, 0.125],
                        [0.375, -0.625, 0.25, 0.5],
                        [0.125, 0.5, -0.25, 0.625],
                    ],
                    dtype=np.float32,
                ),
                bias=np.array([0.125, 0.0, -0.125, 0.25, 0.0, -0.25], dtype=np.float32),
            ),
            "lif_hidden": nir.LIF(
                tau=np.full(6, 20.0),
                r=np.ones(6),
                v_leak=np.zeros(6),
                v_threshold=np.ones(6),
            ),
            "affine_output": nir.Affine(
                weight=np.array(
                    [
                        [0.5, -0.25, 0.125, 0.75, -0.5, 0.25],
                        [-0.125, 0.5, 0.625, -0.25, 0.375, 0.125],
                    ],
                    dtype=np.float32,
                ),
                bias=np.array([0.125, -0.125], dtype=np.float32),
            ),
            "lif_output": nir.LIF(
                tau=np.full(2, 20.0),
                r=np.ones(2),
                v_leak=np.zeros(2),
                v_threshold=np.ones(2),
            ),
            "output": nir.Output(output_type={"output": np.array([2])}),
        },
        edges=[
            ("input", "affine_hidden"),
            ("affine_hidden", "lif_hidden"),
            ("lif_hidden", "affine_output"),
            ("affine_output", "lif_output"),
            ("lif_output", "output"),
        ],
    )


def _build_graph() -> Any:
    """Run the public NIR parser and hardware-graph lowering path."""
    from sc_neurocore.nir_bridge import from_nir, from_scnetwork

    return from_scnetwork(from_nir(_nir_graph(), dt=0.5), dt=0.5)


def _compile(graph: Any) -> Any:
    """Compile one hardware graph through the public FPGA boundary."""
    from sc_neurocore.nir_bridge import compile_network_to_fpga

    return compile_network_to_fpga(
        graph,
        module_name="nir_graph_benchmark",
        data_width=18,
        fraction=10,
        bitstream_length=1024,
    )


def _array(value: np.ndarray[Any, Any] | None) -> list[object] | None:
    """Return a JSON-ready array while preserving an absent optional value."""
    return None if value is None else value.tolist()


def _canonical_payload(graph: Any, result: Any) -> bytes:
    """Serialise all graph and emitted compilation contracts deterministically."""
    from sc_neurocore.ir import scnir_to_dict

    graph_payload = {
        "populations": [
            {
                "name": population.name,
                "neuron_type": population.neuron_type,
                "n_neurons": population.n_neurons,
                "params": {
                    name: values.tolist() for name, values in sorted(population.params.items())
                },
                "dt": population.dt,
            }
            for population in graph.populations
        ],
        "connections": [
            {
                "src": connection.src,
                "dst": connection.dst,
                "weights": connection.weights.tolist(),
                "bias": _array(connection.bias),
                "delay_steps": connection.delay_steps,
                "source_threshold": _array(connection.source_threshold),
                "destination_threshold": _array(connection.destination_threshold),
            }
            for connection in graph.connections
        ],
        "input_pop": graph.input_pop,
        "output_pop": graph.output_pop,
        "dt": graph.dt,
        "hierarchy": [asdict(instance) for instance in graph.hierarchy],
        "summary": graph.summary(),
    }
    compile_payload = {
        "neuron_modules": result.neuron_modules,
        "weight_rom": result.weight_rom,
        "top_module": result.top_module,
        "module_name": result.module_name,
        "total_neurons": result.total_neurons,
        "total_synapses": result.total_synapses,
        "q_format": result.q_format,
        "interconnect": result.interconnect,
        "scnir_document": scnir_to_dict(result.scnir_document),
        "scnir_source_modules": result.scnir_source_modules,
        "scnir_source_manifest": [entry.as_dict() for entry in result.scnir_source_manifest],
        "scnir_external_inputs": [entry.as_dict() for entry in result.scnir_external_inputs],
        "scnir_hierarchy_modules": result.scnir_hierarchy_modules,
        "folded_metrics": None
        if result.folded_metrics is None
        else result.folded_metrics.as_dict(),
        "warnings": result.warnings,
    }
    return json.dumps(
        {"graph": graph_payload, "compile": compile_payload},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _median_latency(operation: Any, repetitions: int) -> tuple[int, Any]:
    """Measure one operation repeatedly and return its median and final value."""
    samples: list[int] = []
    value: Any = None
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        value = operation()
        samples.append(time.perf_counter_ns() - started)
    return int(statistics.median(samples)), value


def main() -> int:
    """Run one probe and write its validated JSON payload to standard output."""
    import_started = time.perf_counter_ns()
    from sc_neurocore.nir_bridge import neuron_graph as graph_module

    import_ns = time.perf_counter_ns() - import_started
    if not hasattr(graph_module, "from_scnetwork"):
        raise RuntimeError("historical neuron-graph module has no conversion surface")
    graph_ns, graph = _median_latency(_build_graph, 25)
    compile_ns, result = _median_latency(lambda: _compile(graph), 5)
    payload = _canonical_payload(graph, result)
    maximum_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        maximum_rss_kib //= 1024
    print(
        json.dumps(
            {
                "import_ns": import_ns,
                "graph_lowering_ns": graph_ns,
                "fpga_compilation_ns": compile_ns,
                "max_rss_kib": maximum_rss_kib,
                "generated_sha256": hashlib.sha256(payload).hexdigest(),
                "generated_bytes": len(payload),
                "population_count": len(graph.populations),
                "connection_count": len(graph.connections),
                "total_neurons": graph.total_neurons,
                "total_synapses": graph.total_synapses,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real snnTorch and NIR inference interchange measurements

"""Measure actual inference paths only after their spike and state checks pass."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import hashlib
from importlib import import_module, metadata
import json
from pathlib import Path
import statistics
import tempfile
import time
from typing import Any

import numpy as np
import snntorch as snn
import torch

from _benchmark_context import load_average, measurement_context
from sc_neurocore.nir_bridge import from_nir, from_snntorch, to_nir, to_snntorch

_MODULES = (
    "__init__",
    "execution_metadata",
    "graph_identity",
    "snntorch_bridge",
    "parser",
    "node_map",
    "export",
    "interchange_notes",
)


def _producer_hashes() -> dict[str, str]:
    result = {}
    for name in _MODULES:
        qualified = "sc_neurocore.nir_bridge" + ("" if name == "__init__" else f".{name}")
        module = import_module(qualified)
        assert module.__file__ is not None
        result[f"sc_neurocore/nir_bridge/{name}.py"] = hashlib.sha256(
            Path(module.__file__).read_bytes()
        ).hexdigest()
    schema = import_module("sc_neurocore.ir.scnir_schema")
    assert schema.__file__ is not None
    result["sc_neurocore/ir/scnir_schema.py"] = hashlib.sha256(
        Path(schema.__file__).read_bytes()
    ).hexdigest()
    return result


def _model(reset: str) -> Any:
    torch.manual_seed(142)
    layers: list[Any] = []
    for before, width, cell_type in [(4, 8, snn.Synaptic), (8, 3, snn.Leaky)]:
        layer = torch.nn.Linear(before, width, dtype=torch.float64)
        with torch.no_grad():
            layer.weight.abs_().mul_(1.3)
            layer.bias.fill_(0.1)
        options = dict(
            beta=torch.linspace(0.5, 0.9, width, dtype=torch.float64),
            threshold=torch.linspace(0.6, 1.1, width, dtype=torch.float64),
            init_hidden=True,
            reset_delay=False,
            reset_mechanism=reset,
        )
        cell = (
            cell_type(alpha=0.7, **options) if cell_type is snn.Synaptic else cell_type(**options)
        )
        cell.graded_spikes_factor = torch.ones(width, dtype=torch.float64)
        layers.extend([layer, cell])
    return torch.nn.Sequential(*layers).double()


def _reset_framework(model: Any) -> None:
    for module in model.modules():
        if isinstance(module, (snn.Leaky, snn.Synaptic)):
            module.reset_mem()


def _framework_trace(model: Any, inputs: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    _reset_framework(model)
    with torch.no_grad():
        return np.stack([model(torch.from_numpy(value)).numpy().copy() for value in inputs])


def _sc_trace(network: Any, inputs: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    network.reset()
    return np.stack(
        [network.step({network.input_nodes[0]: value})[network.output_nodes[0]] for value in inputs]
    )


def _case(
    reset: str, inputs: np.ndarray[Any, Any], iterations: int, warmups: int
) -> dict[str, object]:
    model = _model(reset)
    network = from_snntorch(model, torch.zeros(4, dtype=torch.float64), dt=0.125)
    with tempfile.TemporaryDirectory(prefix="scn-nir-benchmark-") as directory:
        path = Path(directory) / "network.nir"
        to_nir(network, path)
        restored = from_nir(path)
        file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    exported = to_snntorch(restored)
    operations: dict[str, Callable[[], np.ndarray[Any, Any]]] = {
        "source_snntorch": lambda: _framework_trace(model, inputs),
        "sc_reference": lambda: _sc_trace(network, inputs),
        "sc_nir_roundtrip": lambda: _sc_trace(restored, inputs),
        "exported_snntorch": lambda: _framework_trace(exported, inputs),
    }
    expected = operations["source_snntorch"]()
    for operation in operations.values():
        np.testing.assert_array_equal(operation(), expected)
    spikes = int(np.count_nonzero(expected))
    if not 0 < spikes < expected.size:
        raise ValueError("Benchmark must contain spikes and silence")
    sc_cells = [node for node in network.nodes.values() if hasattr(node, "v_threshold")]
    framework_cells = [node for node in model if isinstance(node, (snn.Leaky, snn.Synaptic))]
    final_error = max(
        float(np.max(np.abs(node.v - cell.mem.numpy())))
        for node, cell in zip(sc_cells, framework_cells, strict=True)
    )
    if final_error > 1e-12:
        raise ValueError("Framework membrane states disagree before timing")
    names = list(operations)
    for _ in range(warmups):
        for operation in operations.values():
            operation()
    samples: dict[str, list[int]] = {name: [] for name in names}
    for repetition in range(iterations):
        # Alternate the path order so one implementation is not always first.
        for name in names if repetition % 2 == 0 else reversed(names):
            started = time.perf_counter_ns()
            actual = operations[name]()
            samples[name].append(time.perf_counter_ns() - started)
            np.testing.assert_array_equal(actual, expected)
    return {
        "framework_reset": reset,
        "sc_reset": network.reset_mode,
        "timestep": network.dt,
        "spikes_per_trace": spikes,
        "final_membrane_max_abs_error": final_error,
        "spikes_identical": True,
        "nir_file_sha256": file_digest,
        "trace_sha256": hashlib.sha256(expected.astype(np.float64).tobytes()).hexdigest(),
        "paths": {
            name: {
                "elapsed_ns": values,
                "median_ns_per_step": statistics.median(values) / len(inputs),
            }
            for name, values in samples.items()
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run real inference comparisons and write source-bound measurement evidence.

    Parameters
    ----------
    argv : sequence of str or None
        Iteration, warmup, timestep-count and JSON output options.

    Returns
    -------
    int
        Zero only after every measured trace matches the actual source framework.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.iterations < 1 or args.warmups < 0 or args.steps < 8:
        parser.error("iterations must be positive, warmups nonnegative, and steps at least eight")
    torch.set_num_threads(1)
    inputs = np.random.default_rng(54).uniform(-0.4, 2.0, (args.steps, 4))
    source_before = _producer_hashes()
    script_before = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    load_before = load_average()
    cases = [_case(reset, inputs, args.iterations, args.warmups) for reset in ("zero", "subtract")]
    if (
        source_before != _producer_hashes()
        or script_before != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    ):
        raise RuntimeError("Interchange producers changed during measurement")
    payload = {
        "SPDX-License-Identifier": "AGPL-3.0-or-later",
        "schema_version": "sc-neurocore.nir-interoperability-benchmark.v1",
        "evidence_class": "local_inference_regression",
        "hardware_execution_claimed": False,
        "configuration": {
            "iterations": args.iterations,
            "warmups": args.warmups,
            "steps": args.steps,
            "dtype": "float64",
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
            "seed": 54,
        },
        "versions": {
            name: metadata.version(name)
            for name in ("numpy", "torch", "snntorch", "nir", "nirtorch")
        },
        "benchmark_sha256": script_before,
        "source_hashes": source_before,
        "cases": cases,
        "measurement_context": measurement_context(load_before),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Real NIR/framework inference evidence written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

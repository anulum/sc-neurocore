# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public NIR lowering benchmark provenance

"""Run the real benchmark command and verify its measured producer bindings."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "src/sc_neurocore/nir_bridge"
BENCHMARK_PATH = ROOT / "benchmarks/bench_nir_graph.py"
PROBE_PATH = ROOT / "benchmarks/_nir_graph_probe.py"
BENCHMARK_RESULT_PATH = ROOT / "benchmarks/results/bench_nir_graph.json"


def test_benchmark_command_binds_parser_and_factory_sources(tmp_path: Path) -> None:
    """Bind actual cold-process lowering and FPGA output to its imported sources."""
    output = tmp_path / "lowering.json"
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks/bench_nir_graph.py"),
            "--baseline-root",
            str(ROOT),
            "--baseline-ref",
            "same-source",
            "--iterations",
            "1",
            "--warmups",
            "0",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    report = json.loads(output.read_text())
    assert report["comparison"]["generated_output_byte_identical"] is True
    assert report["comparison"]["total_neurons"] == 8
    assert report["comparison"]["total_synapses"] == 36
    assert report["results"]["candidate"]["sample_count"] == 1
    assert report["results"]["candidate"]["graph_lowering_ns"]["median"] > 0
    for name in (
        "__init__",
        "parser",
        "node_map",
        "execution_metadata",
        "graph_identity",
        "interchange_notes",
    ):
        relative = f"src/sc_neurocore/nir_bridge/{name}.py"
        assert (
            report["source_hashes"][relative]
            == hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        )
    assert report["source_file_count"] == len(report["source_hashes"]) == 14


def test_benchmark_evidence_is_source_bound_and_byte_identical() -> None:
    """Pin the measured parent/candidate fidelity result to its live producers."""
    payload = json.loads(BENCHMARK_RESULT_PATH.read_text(encoding="utf-8"))
    source_paths = sorted(
        set(SOURCE_DIR.glob("neuron_graph*.py"))
        | {
            SOURCE_DIR / f"{name}.py"
            for name in (
                "__init__",
                "parser",
                "node_map",
                "execution_metadata",
                "graph_identity",
                "interchange_notes",
            )
        }
    )
    expected_hashes = {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths
    }

    assert payload["benchmark_sha256"] == hashlib.sha256(BENCHMARK_PATH.read_bytes()).hexdigest()
    assert payload["probe_sha256"] == hashlib.sha256(PROBE_PATH.read_bytes()).hexdigest()
    assert payload["source_file_count"] == len(expected_hashes)
    assert payload["source_hashes"] == expected_hashes
    assert payload["configuration"]["iterations"] == 30
    assert payload["configuration"]["warmups"] == 2
    assert payload["results"]["parent"]["sample_count"] == 30
    assert payload["results"]["candidate"]["sample_count"] == 30
    expected_comparison = {
        "generated_output_byte_identical": True,
        "generated_sha256": "d1a19a26a33912f5c10c29d2f0d86cb14693d90a6a8242b8c1b063e6cf3bd899",
        "generated_bytes": 22963,
        "population_count": 2,
        "connection_count": 2,
        "total_neurons": 8,
        "total_synapses": 36,
    }
    assert {key: payload["comparison"][key] for key in expected_comparison} == expected_comparison

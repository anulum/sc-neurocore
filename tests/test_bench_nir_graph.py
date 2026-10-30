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

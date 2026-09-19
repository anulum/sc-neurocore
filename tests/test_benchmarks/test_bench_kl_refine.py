# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — KL-refinement benchmark tests

"""Source binding and evidence-shape tests for the KL benchmark."""

from __future__ import annotations

import importlib
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from tools.benchmark_evidence_gate import committed_source_hash_failures


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))
benchmark: Any = importlib.import_module("bench_kl_refine")


def test_source_manifest_binds_every_maintained_implementation() -> None:
    """The result identifies Python sources and every real backend kernel."""
    manifest = benchmark._source_manifest()
    assert len(manifest["combined_source_sha256"]) == 64
    assert len(manifest["runner_sha256"]) == 64
    sources = manifest["files"]
    assert "engine/src/partition.rs" in sources
    assert "src/sc_neurocore/accel/julia/chiplet/kl_refine.jl" in sources
    assert "src/sc_neurocore/accel/go/partition/partition.go" in sources
    assert "src/sc_neurocore/accel/mojo/partition/partition.mojo" in sources
    assert all(len(digest) == 64 for digest in sources.values())


def test_workload_builder_and_initial_partition_are_deterministic() -> None:
    """Repeated construction produces the same graph and vertex membership."""
    first = benchmark._build_graph(20, deg=4, seed=7)
    second = benchmark._build_graph(20, deg=4, seed=7)
    assert first.edges == second.edges
    partitions = benchmark._initial_partitions(20, 4)
    assert sorted(vertex for part in partitions for vertex in part) == list(range(20))


def test_environment_declares_diagnostic_timing_class() -> None:
    """Host timing cannot be mistaken for an isolated promotion result."""
    environment = benchmark._environment()
    assert environment["isolated_cpu_claim"] is False
    assert environment["timing_class"] == "loaded-host diagnostic"
    assert isinstance(environment["affinity"], list)


def test_committed_result_is_source_bound_and_parity_gated() -> None:
    """The checked-in result matches measured source and parity evidence."""
    artifact = ROOT / "benchmarks/results/bench_kl_refine.json"
    result = json.loads(artifact.read_text(encoding="utf-8"))
    manifest = result["source_manifest"]
    sources = manifest["files"]
    assert result["schema_version"] == benchmark.BENCHMARK_SCHEMA_VERSION
    assert result["label"] == "candidate"
    assert result["canonical_equivalence"] is True
    assert len(result["canonical_partition_sha256"]) == 64
    assert not committed_source_hash_failures(
        artifact,
        repo_root=ROOT,
        source_hash_paths={
            **{path: f"source_manifest.files.{path}" for path in sources},
            "benchmarks/bench_kl_refine.py": "source_manifest.runner_sha256",
        },
    )
    digest = hashlib.sha256()
    for path, source_hash in sorted(sources.items()):
        digest.update(path.encode("utf-8") + b"\0" + source_hash.encode("ascii") + b"\n")
    assert manifest["combined_source_sha256"] == digest.hexdigest()
    assert result["gate_source_hashes"] == benchmark._gate_source_hashes(manifest)
    assert all(info["available"] is True for info in result["backends"].values())
    assert all(row["parity_ok"] is True for row in result["rows"])
    assert set(result["timings_ms"]) == {"v100", "v200", "v500", "v1000"}
    assert all(
        set(timings) == {"python", "rust", "julia", "go", "mojo"}
        for timings in result["timings_ms"].values()
    )

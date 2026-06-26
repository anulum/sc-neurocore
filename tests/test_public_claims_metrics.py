# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_capability_manifest() -> ModuleType:
    tool_path = _repo_root() / "tools" / "capability_manifest.py"
    spec = importlib.util.spec_from_file_location("capability_manifest", tool_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _manifest_counts() -> dict[str, int]:
    tool = _load_capability_manifest()
    manifest: dict[str, Any] = tool.build_capability_manifest(_repo_root())
    counts = manifest["counts"]
    return {
        "python_model_source_modules": int(counts["python_model_source_modules"]),
        "python_model_classes": int(counts["python_model_classes"]),
        "rust_pyo3_model_wrappers": int(counts["rust_pyo3_model_wrappers"]),
        "test_files": int(counts["test_files"]),
        "public_documentation_pages": int(counts["public_documentation_pages"]),
    }


def _rust_networkrunner_model_count() -> int:
    source = (_repo_root() / "engine/src/network_runner.rs").read_text(encoding="utf-8")
    match = re.search(
        r"pub\s+fn\s+supported_models\s*\([^)]*\)\s*->\s*[^{]+\{(?P<body>.*?)\n\s*\}",
        source,
        re.DOTALL,
    )
    assert match is not None
    return len(re.findall(r'"[A-Za-z0-9_]+"', match.group("body")))


def _formal_inventory() -> tuple[int, dict[str, int]]:
    formal_root = _repo_root() / "hdl/formal"
    statements = {"assert": 0, "assume": 0, "cover": 0}
    formal_sources = [
        path
        for extension in ("*.v", "*.sv", "*.svh")
        for path in formal_root.rglob(extension)
    ]
    for path in sorted(
        formal_sources,
        key=lambda candidate: candidate.relative_to(formal_root).as_posix(),
    ):
        text = path.read_text(encoding="utf-8")
        for keyword in statements:
            statements[keyword] += len(
                re.findall(rf"\b{keyword}\s*(?:property)?\s*\(", text)
            )
    return len(list(formal_root.rglob("*.sby"))), statements


def _compact_whitespace(text: str) -> str:
    return " ".join(text.split())


def test_public_capability_snapshots_are_current() -> None:
    tool = _load_capability_manifest()

    tool.assert_outputs_current(_repo_root())


def test_current_public_entrypoints_use_generated_inventory_terms() -> None:
    counts = _manifest_counts()
    rust_models = _rust_networkrunner_model_count()
    readme = (_repo_root() / "README.md").read_text(encoding="utf-8")
    docs_index = (_repo_root() / "docs/index.md").read_text(encoding="utf-8")
    neuron_reference = (_repo_root() / "docs/api/neuron_models.md").read_text(encoding="utf-8")
    all_entrypoints = "\n".join((readme, docs_index, neuron_reference))

    assert (
        f"{counts['python_model_source_modules']} Python model source modules"
        in all_entrypoints
    )
    assert f"{counts['python_model_classes']} lazy-loaded Python model classes" in all_entrypoints
    assert f"{counts['rust_pyo3_model_wrappers']} Rust PyO3 model wrappers" in all_entrypoints
    assert f"{rust_models}-model NetworkRunner" in all_entrypoints
    assert f"| Python test files | {counts['test_files']} |" in readme

    stale_claims = (
        "174 neuron models",
        "174 Rust neuron models",
        "173 Neuron Models",
        "173 Rust implementations",
        "121 Python",
        "120**",
        "72 properties",
    )
    for stale_claim in stale_claims:
        assert stale_claim not in all_entrypoints


def test_public_formal_claims_match_hdl_inventory() -> None:
    proof_jobs, statements = _formal_inventory()
    total = sum(statements.values())
    expected_summary = (
        f"{proof_jobs} SymbiYosys proof jobs and {total} formal statements "
        f"({statements['assert']} assert, {statements['assume']} assume, "
        f"{statements['cover']} cover)"
    )

    readme = (_repo_root() / "README.md").read_text(encoding="utf-8")
    comparison = (_repo_root() / "docs/benchmarks/comparison.md").read_text(encoding="utf-8")
    tutorial = (_repo_root() / "docs/tutorials/21_formal_verification.md").read_text(
        encoding="utf-8"
    )

    assert expected_summary in _compact_whitespace(readme)
    assert expected_summary in _compact_whitespace(comparison)
    assert expected_summary in _compact_whitespace(tutorial)


def test_rust_speedup_claims_are_artifact_anchored() -> None:
    readme = (_repo_root() / "README.md").read_text(encoding="utf-8")
    faq = (_repo_root() / "docs/guides/faq.md").read_text(encoding="utf-8")
    landscape = (_repo_root() / "docs/COMPETITIVE_LANDSCAPE.md").read_text(encoding="utf-8")

    speedup_evidence = "benchmarks/results/rust_scaling_benchmark.json"
    assert speedup_evidence in readme
    assert speedup_evidence in faq
    assert speedup_evidence in landscape
    assert "Brunel balanced-network" in readme
    assert "39-202x faster" not in readme
    assert "39–202× faster" not in readme

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools/audit_model_docs.py"


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("audit_model_docs", TOOL)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_manifest_counts_doc_tests_and_benchmark_evidence(tmp_path: Path) -> None:
    tool = _load_tool()
    repo = tmp_path
    _write(
        repo / "src/sc_neurocore/neurons/models/example_model.py",
        '"""Example source."""\n\n'
        "class ExampleModel:\n"
        "    def __init__(self, tau: float = 10.0, threshold: float = 1.0):\n"
        "        pass\n",
    )
    _write(
        repo / "docs/api/models/example_model.md",
        "# Example Model\n\n"
        "Source: src/sc_neurocore/neurons/models/example_model.py\n\n"
        "## Equation\nDynamics update rule.\n\n"
        "## Parameters\nParameter table.\n\n"
        "## Defaults\nDefault values.\n\n"
        "## Benchmarks\nPerformance evidence.\n\n"
        "## Verification\nValidated by tests.\n",
    )
    _write(repo / "tests/test_model_example_model.py", "def test_example_model():\n    pass\n")
    _write(repo / "benchmarks/results/bench_example_model.json", "{}\n")

    manifest = tool.build_manifest(repo, timestamp="2026-04-30T120000Z")

    assert manifest["schema_version"] == tool.SCHEMA_VERSION
    assert manifest["counts"]["source_models"] == 1
    assert manifest["counts"]["doc_pages"] == 1
    assert manifest["counts"]["status"] == {"PASS": 1}
    entry = manifest["entries"][0]
    assert entry["source"]["classes"] == ["ExampleModel"]
    assert entry["source"]["init_parameters"] == {"ExampleModel": ["tau", "threshold"]}
    assert entry["status"] == "PASS"
    assert entry["missing"] == []


def test_manifest_surfaces_missing_evidence_and_orphan_docs(tmp_path: Path) -> None:
    tool = _load_tool()
    repo = tmp_path
    _write(
        repo / "src/sc_neurocore/neurons/models/thin_model.py",
        "class ThinModel:\n    pass\n",
    )
    _write(repo / "docs/api/models/thin_model.md", "# Thin Model\n\nParameters only.\n")
    _write(repo / "docs/api/models/unmapped_network.md", "# Unmapped Network\n")

    manifest = tool.build_manifest(repo, timestamp="2026-04-30T120000Z")
    entry = manifest["entries"][0]

    assert entry["status"] == "NEEDS_TEST"
    assert "has_tests" in entry["missing"]
    assert "has_benchmark_artifact" in entry["missing"]
    assert "docs/api/models/unmapped_network.md" in manifest["orphan_doc_pages"]


def test_markdown_report_keeps_machine_and_human_gates_separate(tmp_path: Path) -> None:
    tool = _load_tool()
    manifest = {
        "generated_at": "2026-04-30T120000Z",
        "counts": {
            "source_models": 1,
            "doc_pages": 1,
            "orphan_doc_pages": 0,
            "status": {"NEEDS_TEST": 1},
        },
        "entries": [
            {
                "model": "thin_model",
                "status": "NEEDS_TEST",
                "missing": ["has_tests"],
                "doc": {"path": "docs/api/models/thin_model.md"},
                "tests": [],
                "benchmark_artifacts": [],
            }
        ],
        "orphan_doc_pages": [],
    }

    report = tool.render_markdown(manifest)

    assert "append-only audit evidence" in report
    assert "biological interpretation before status promotion" in report
    assert "`thin_model`" in report


def test_outputs_are_timestamped(tmp_path: Path) -> None:
    tool = _load_tool()
    manifest = {
        "generated_at": "2026-04-30T120000Z",
        "counts": {
            "source_models": 0,
            "doc_pages": 0,
            "orphan_doc_pages": 0,
            "status": {},
        },
        "entries": [],
        "orphan_doc_pages": [],
    }

    json_path, md_path = tool.write_manifest_outputs(
        manifest,
        tmp_path,
        "2026-04-30T120000Z",
    )

    assert json_path.name == "model_doc_audit_2026_04_30T120000Z.json"
    assert md_path.name == "model_doc_audit_2026_04_30T120000Z.md"
    assert json_path.exists()
    assert md_path.exists()


def test_review_batch_filters_status_and_limits_entries() -> None:
    tool = _load_tool()
    manifest = {
        "generated_at": "2026-04-30T120000Z",
        "entries": [
            {
                "model": "needs_benchmark_model",
                "status": "NEEDS_BENCHMARK",
                "missing": ["has_benchmark_artifact"],
                "doc": {"path": "docs/api/models/needs_benchmark_model.md"},
            },
            {
                "model": "needs_test_model",
                "status": "NEEDS_TEST",
                "missing": ["has_tests"],
                "doc": {"path": "docs/api/models/needs_test_model.md"},
            },
            {
                "model": "pass_model",
                "status": "PASS",
                "missing": [],
                "doc": {"path": "docs/api/models/pass_model.md"},
            },
        ],
    }

    report = tool.render_review_batch(manifest, statuses=["needs_test"], limit=1)

    assert "Filter: `NEEDS_TEST`" in report
    assert "`needs_test_model`" in report
    assert "`needs_benchmark_model`" not in report
    assert "Add or map behavioural/parity tests before prose promotion." in report


def test_review_batch_filters_missing_evidence_key() -> None:
    tool = _load_tool()
    manifest = {
        "generated_at": "2026-04-30T120000Z",
        "entries": [
            {
                "model": "missing_source_link",
                "status": "NEEDS_TEST",
                "missing": ["has_source_link", "has_tests"],
                "doc": {"path": "docs/api/models/missing_source_link.md"},
            },
            {
                "model": "missing_tests_only",
                "status": "NEEDS_TEST",
                "missing": ["has_tests"],
                "doc": {"path": "docs/api/models/missing_tests_only.md"},
            },
        ],
    }

    report = tool.render_review_batch(manifest, missing=["has_source_link"], limit=10)

    assert "Missing evidence: `has_source_link`" in report
    assert "`missing_source_link`" in report
    assert "`missing_tests_only`" not in report


def test_write_review_batch_output_uses_status_slug(tmp_path: Path) -> None:
    tool = _load_tool()
    manifest = {
        "generated_at": "2026-04-30T120000Z",
        "entries": [
            {
                "model": "thin_model",
                "status": "NEEDS_DOC_EVIDENCE",
                "missing": ["has_equation_section"],
                "doc": {"path": "docs/api/models/thin_model.md"},
            }
        ],
    }

    path = tool.write_review_batch_output(
        manifest,
        tmp_path,
        "2026-04-30T120000Z",
        statuses=["NEEDS_DOC_EVIDENCE"],
        missing=["has_equation_section"],
        limit=10,
    )

    assert (
        path.name == "model_doc_batch_2026_04_30T120000Z_needs_doc_evidence_has_equation_section.md"
    )
    assert "`thin_model`" in path.read_text()


def test_review_batch_rejects_unknown_status() -> None:
    tool = _load_tool()
    manifest = {"generated_at": "2026-04-30T120000Z", "entries": []}

    try:
        tool.render_review_batch(manifest, statuses=["MISSING"], limit=10)
    except Exception as exc:
        assert "unknown audit status" in str(exc)
    else:
        raise AssertionError("expected unknown status to fail")


def test_review_batch_rejects_unknown_missing_key() -> None:
    tool = _load_tool()
    manifest = {"generated_at": "2026-04-30T120000Z", "entries": []}

    try:
        tool.render_review_batch(manifest, missing=["not_a_rubric_key"], limit=10)
    except Exception as exc:
        assert "unknown missing-evidence key" in str(exc)
    else:
        raise AssertionError("expected unknown missing key to fail")

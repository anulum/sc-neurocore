# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Hold the benchmark evidence gate to real records, sources and Git history."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


def _load_tool() -> Any:
    repo_root = Path(__file__).resolve().parents[2]
    tool_path = repo_root / "tools" / "benchmark_evidence_gate.py"
    spec = importlib.util.spec_from_file_location("benchmark_evidence_gate", tool_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _commit(repo: Path, message: str) -> None:
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-qm", message],
        check=True,
        capture_output=True,
    )


def _historical_repo(repo: Path, *, digest: str | None = None) -> tuple[Path, Path]:
    tool = _load_tool()
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Benchmark Test"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "benchmark@example.invalid"],
        check=True,
        capture_output=True,
    )
    source = repo / "benchmarks" / "bench_demo.py"
    source.parent.mkdir(parents=True)
    source.write_text("print('measured')\n", encoding="utf-8")
    artefact = repo / "benchmarks" / "results" / "demo.json"
    _write_json(
        artefact,
        {
            "latency_ns": 10.0,
            "source_hashes": {"benchmarks/bench_demo.py": digest or tool._sha256(source)},
        },
    )
    manifest = repo / "benchmarks" / "benchmark_regression_gates.json"
    _write_json(
        manifest,
        {
            "SPDX-License-Identifier": "AGPL-3.0-or-later",
            "schema_version": "sc-neurocore.benchmark-regression-gates.v1",
            "source_hash_policy": "artefact_commit",
            "gates": [
                {
                    "id": "demo",
                    "artefact": "benchmarks/results/demo.json",
                    "required_numbers": ["latency_ns"],
                    "source_hashes": {
                        "benchmarks/bench_demo.py": "source_hashes.benchmarks/bench_demo.py"
                    },
                }
            ],
        },
    )
    _commit(repo, "record benchmark")
    return artefact, manifest


def test_historical_result_survives_later_source_change(tmp_path: Path) -> None:
    """The real gate binds a measurement to its source revision, not HEAD."""
    tool = _load_tool()
    repo = tmp_path / "repository"
    artefact, manifest = _historical_repo(repo)
    source = repo / "benchmarks" / "bench_demo.py"
    source.write_text("print('changed')\n", encoding="utf-8")
    _commit(repo, "change source after measurement")

    report = tool.evaluate_benchmark_evidence_gate(
        manifest_path=manifest,
        output_path=tmp_path / "report.json",
        repo_root=repo,
    )

    assert report["passed"] is True
    assert report["source_hash_policy"] == "artefact_commit"
    assert tool.committed_source_hash_failures(artefact, repo_root=repo) == []


def test_historical_result_rejects_forged_digest_and_edited_record(tmp_path: Path) -> None:
    """A historical policy still refuses a forged or uncommitted result."""
    tool = _load_tool()
    repo = tmp_path / "repository"
    artefact, manifest = _historical_repo(repo, digest="0" * 64)

    forged = tool.evaluate_benchmark_evidence_gate(
        manifest_path=manifest,
        output_path=tmp_path / "forged.json",
        repo_root=repo,
    )
    assert [row["reason"] for row in forged["failures"]] == ["source_hash_mismatch"]

    payload = json.loads(artefact.read_text(encoding="utf-8"))
    payload["source_hashes"]["benchmarks/bench_demo.py"] = tool._sha256(
        repo / "benchmarks" / "bench_demo.py"
    )
    _write_json(artefact, payload)
    edited = tool.evaluate_benchmark_evidence_gate(
        manifest_path=manifest,
        output_path=tmp_path / "edited.json",
        repo_root=repo,
    )
    assert [row["reason"] for row in edited["failures"]] == ["artefact_differs_from_commit"]


def test_historical_result_rejects_missing_history_and_source(tmp_path: Path) -> None:
    """A shallow checkout or absent historical source cannot establish provenance."""
    tool = _load_tool()
    repo = tmp_path / "repository"
    artefact, manifest = _historical_repo(repo)
    source = repo / "benchmarks" / "bench_demo.py"
    source.write_text("print('later')\n", encoding="utf-8")
    _commit(repo, "change source")

    shallow = tmp_path / "shallow"
    subprocess.run(
        ["git", "clone", "-q", "--depth", "1", repo.as_uri(), str(shallow)],
        check=True,
        capture_output=True,
    )
    unavailable = tool.evaluate_benchmark_evidence_gate(
        manifest_path=shallow / "benchmarks" / "benchmark_regression_gates.json",
        output_path=tmp_path / "shallow-report.json",
        repo_root=shallow,
    )
    assert [row["reason"] for row in unavailable["failures"]] == ["history_is_shallow"]

    payload = json.loads(artefact.read_text(encoding="utf-8"))
    payload["source_hashes"] = {"benchmarks/missing.py": "0" * 64}
    _write_json(artefact, payload)
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["gates"][0]["source_hashes"] = {
        "benchmarks/missing.py": "source_hashes.benchmarks/missing.py"
    }
    _write_json(manifest, manifest_payload)
    _commit(repo, "record invalid source reference")
    missing = tool.evaluate_benchmark_evidence_gate(
        manifest_path=manifest,
        output_path=tmp_path / "missing-report.json",
        repo_root=repo,
    )
    assert [row["reason"] for row in missing["failures"]] == ["missing_historical_source"]


def test_historical_result_accepts_lane_keyed_source_digest(tmp_path: Path) -> None:
    """Explicit JSON-path mapping binds a lane-keyed digest to its source."""
    tool = _load_tool()
    repo = tmp_path / "repository"
    artefact, _ = _historical_repo(repo)
    source = repo / "benchmarks" / "bench_demo.py"
    payload = {"source_hashes": {"python": tool._sha256(source)}}
    _write_json(artefact, payload)
    _commit(repo, "record lane keyed evidence")
    paths = {"benchmarks/bench_demo.py": "source_hashes.python"}

    assert (
        tool.committed_source_hash_failures(artefact, repo_root=repo, source_hash_paths=paths) == []
    )

    payload["source_hashes"]["python"] = "0" * 64
    _write_json(artefact, payload)
    _commit(repo, "forge lane digest")
    failures = tool.committed_source_hash_failures(
        artefact, repo_root=repo, source_hash_paths=paths
    )
    assert [failure.reason for failure in failures] == ["source_hash_mismatch"]


def test_record_is_current_while_its_source_is_unchanged(tmp_path: Path) -> None:
    """A record whose source still has the recorded digest is current."""
    tool = _load_tool()
    repo = tmp_path / "repository"
    artefact, _manifest = _historical_repo(repo)

    assert tool.current_source_hash_failures(artefact, repo_root=repo) == []


def test_changed_source_is_named_while_the_committed_proof_still_holds(tmp_path: Path) -> None:
    """After a source change the record stays proven and stops being current."""
    tool = _load_tool()
    repo = tmp_path / "repository"
    artefact, _manifest = _historical_repo(repo)
    source = repo / "benchmarks" / "bench_demo.py"
    source.write_text("print('changed')\n", encoding="utf-8")
    _commit(repo, "change source after measurement")

    assert tool.committed_source_hash_failures(artefact, repo_root=repo) == []
    failures = tool.current_source_hash_failures(artefact, repo_root=repo)
    assert [failure.reason for failure in failures] == ["source_hash_mismatch"]
    assert failures[0].path.split("->")[0] == "benchmarks/bench_demo.py"

    source.unlink()
    failures = tool.current_source_hash_failures(artefact, repo_root=repo)
    assert [failure.reason for failure in failures] == ["missing_source_for_hash_check"]
    assert failures[0].path == "benchmarks/bench_demo.py"


def _tracked_and_untracked_tree(repo: Path) -> None:
    """Make a repository with one tracked record, one untracked and one ignored."""
    results = repo / "benchmarks" / "results"
    (results / "local").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
    (repo / ".gitignore").write_text("benchmarks/results/local/\n", encoding="utf-8")
    _write_json(results / "tracked.json", {"kept": True})
    subprocess.run(
        ["git", "-C", str(repo), "add", ".gitignore", "benchmarks/results/tracked.json"],
        check=True,
        capture_output=True,
    )
    _write_json(results / "untracked.json", {"kept": False})
    _write_json(results / "local" / "ignored.json", {"kept": False})


def test_tracked_files_lists_only_what_git_tracks(tmp_path: Path) -> None:
    """An untracked file and an ignored file are present on disk and are not listed."""
    tool = _load_tool()
    repo = tmp_path / "repo"
    _tracked_and_untracked_tree(repo)
    assert (repo / "benchmarks" / "results" / "untracked.json").is_file()
    assert (repo / "benchmarks" / "results" / "local" / "ignored.json").is_file()
    assert tool.tracked_files(repo, "benchmarks/results") == ("benchmarks/results/tracked.json",)
    assert tool.tracked_files(repo, "benchmarks/absent") == ()


def test_tracked_files_is_none_where_git_cannot_say(tmp_path: Path) -> None:
    """Outside a work tree, and below the top of one, the caller has to walk the directory."""
    tool = _load_tool()
    plain = tmp_path / "plain"
    (plain / "benchmarks" / "results").mkdir(parents=True)
    assert tool.tracked_files(plain, "benchmarks/results") is None
    repo = tmp_path / "repo"
    _tracked_and_untracked_tree(repo)
    assert tool.tracked_files(repo / "benchmarks", "results") is None


def test_tracked_files_is_none_when_the_index_cannot_be_read(tmp_path: Path) -> None:
    """A work tree whose index is damaged gives no listing instead of an empty one."""
    tool = _load_tool()
    repo = tmp_path / "repo"
    _tracked_and_untracked_tree(repo)
    (repo / ".git" / "index").write_bytes(b"not an index")
    assert tool.tracked_files(repo, "benchmarks/results") is None


def test_tracked_files_is_none_without_a_git_program(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Where no git program can be started the caller walks the directory, and nothing is raised."""
    tool = _load_tool()
    repo = tmp_path / "repo"
    _tracked_and_untracked_tree(repo)
    empty = tmp_path / "no-programs"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    assert tool.tracked_files(repo, "benchmarks/results") is None


def test_record_without_a_path_keyed_digest_binds_no_source(tmp_path: Path) -> None:
    """Digests keyed by a lane name, or no digests, bind no file by path."""
    tool = _load_tool()
    repo = tmp_path / "repository"
    artefact, _manifest = _historical_repo(repo)

    _write_json(artefact, {"latency_ns": 10.0, "source_hashes": {"python": "0" * 64}})
    failures = tool.current_source_hash_failures(artefact, repo_root=repo)
    assert [failure.reason for failure in failures] == ["missing_path_keyed_source_hashes"]

    _write_json(artefact, {"latency_ns": 10.0})
    failures = tool.current_source_hash_failures(artefact, repo_root=repo)
    assert [failure.reason for failure in failures] == ["missing_source_hashes"]

    paths = {"benchmarks/bench_demo.py": "source_hashes.python"}
    _write_json(
        artefact,
        {
            "latency_ns": 10.0,
            "source_hashes": {"python": tool._sha256(repo / "benchmarks" / "bench_demo.py")},
        },
    )
    assert (
        tool.current_source_hash_failures(artefact, repo_root=repo, source_hash_paths=paths) == []
    )

    artefact.unlink()
    failures = tool.current_source_hash_failures(artefact, repo_root=repo)
    assert [failure.reason for failure in failures] == ["artefact_is_missing_or_invalid"]


def test_benchmark_evidence_gate_accepts_reviewed_manifest(tmp_path: Path) -> None:
    """A manifest whose artefact, numbers and source digest agree passes."""
    tool = _load_tool()
    source = tmp_path / "benchmarks" / "bench_demo.py"
    source.parent.mkdir(parents=True)
    source.write_text("print('bench')\n", encoding="utf-8")
    artefact = tmp_path / "benchmarks" / "results" / "demo.json"
    source_hash = tool._sha256(source)
    _write_json(
        artefact,
        {
            "latency_ns": 10.0,
            "throughput_hz": 200.0,
            "passed": True,
            "source_sha256": source_hash,
        },
    )
    manifest = tmp_path / "benchmarks" / "benchmark_regression_gates.json"
    _write_json(
        manifest,
        {
            "SPDX-License-Identifier": "AGPL-3.0-or-later",
            "schema_version": "sc-neurocore.benchmark-regression-gates.v1",
            "gates": [
                {
                    "id": "demo",
                    "artefact": "benchmarks/results/demo.json",
                    "required_numbers": ["latency_ns", "throughput_hz"],
                    "expected_values": {"passed": True},
                    "source_hashes": {"benchmarks/bench_demo.py": "source_sha256"},
                    "regression_limits": {"latency_ns": {"max": 15.0}},
                }
            ],
        },
    )
    output = tmp_path / "benchmarks" / "results" / "gate.json"

    report = tool.evaluate_benchmark_evidence_gate(
        manifest_path=manifest,
        output_path=output,
        repo_root=tmp_path,
    )

    assert report["passed"] is True
    assert report["failure_count"] == 0
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is True


def test_benchmark_evidence_gate_fails_closed_on_missing_metric(tmp_path: Path) -> None:
    """A required number that is absent from the artefact fails the gate."""
    tool = _load_tool()
    _write_json(tmp_path / "benchmarks" / "results" / "demo.json", {"latency_ns": 10.0})
    manifest = tmp_path / "benchmarks" / "benchmark_regression_gates.json"
    _write_json(
        manifest,
        {
            "SPDX-License-Identifier": "AGPL-3.0-or-later",
            "schema_version": "sc-neurocore.benchmark-regression-gates.v1",
            "gates": [
                {
                    "id": "demo",
                    "artefact": "benchmarks/results/demo.json",
                    "required_numbers": ["latency_ns", "throughput_hz"],
                }
            ],
        },
    )

    report = tool.evaluate_benchmark_evidence_gate(
        manifest_path=manifest,
        output_path=tmp_path / "gate.json",
        repo_root=tmp_path,
    )

    assert report["passed"] is False
    assert report["failures"] == [
        {
            "gate_id": "demo",
            "path": "throughput_hz",
            "reason": "missing_required_numeric_metric",
        }
    ]


def test_benchmark_evidence_gate_rejects_stale_source_hash(tmp_path: Path) -> None:
    """A digest that no longer equals its source fails the gate."""
    tool = _load_tool()
    source = tmp_path / "benchmarks" / "bench_demo.py"
    source.parent.mkdir(parents=True)
    source.write_text("print('new')\n", encoding="utf-8")
    _write_json(
        tmp_path / "benchmarks" / "results" / "demo.json",
        {"source_sha256": "0" * 64, "latency_ns": 1.0},
    )
    manifest = tmp_path / "benchmarks" / "benchmark_regression_gates.json"
    _write_json(
        manifest,
        {
            "SPDX-License-Identifier": "AGPL-3.0-or-later",
            "schema_version": "sc-neurocore.benchmark-regression-gates.v1",
            "gates": [
                {
                    "id": "demo",
                    "artefact": "benchmarks/results/demo.json",
                    "required_numbers": ["latency_ns"],
                    "source_hashes": {"benchmarks/bench_demo.py": "source_sha256"},
                }
            ],
        },
    )

    report = tool.evaluate_benchmark_evidence_gate(
        manifest_path=manifest,
        output_path=tmp_path / "gate.json",
        repo_root=tmp_path,
    )

    assert report["passed"] is False
    assert report["failures"][0]["reason"] == "source_hash_mismatch"


def test_benchmark_evidence_gate_accepts_dotted_source_filename(tmp_path: Path) -> None:
    """Flat source-hash keys retain filename suffixes such as ``.py``."""
    tool = _load_tool()
    source = tmp_path / "benchmarks" / "bench_demo.py"
    source.parent.mkdir(parents=True)
    source.write_text("print('bench')\n", encoding="utf-8")
    source_hash = tool._sha256(source)
    _write_json(
        tmp_path / "benchmarks" / "results" / "demo.json",
        {"latency_ns": 1.0, "source_hashes": {"benchmarks/bench_demo.py": source_hash}},
    )
    manifest = tmp_path / "benchmarks" / "benchmark_regression_gates.json"
    _write_json(
        manifest,
        {
            "SPDX-License-Identifier": "AGPL-3.0-or-later",
            "schema_version": "sc-neurocore.benchmark-regression-gates.v1",
            "gates": [
                {
                    "id": "dotted-source-filename",
                    "artefact": "benchmarks/results/demo.json",
                    "required_numbers": ["latency_ns"],
                    "source_hashes": {
                        "benchmarks/bench_demo.py": "source_hashes.benchmarks/bench_demo.py"
                    },
                }
            ],
        },
    )

    report = tool.evaluate_benchmark_evidence_gate(
        manifest_path=manifest,
        output_path=tmp_path / "gate.json",
        repo_root=tmp_path,
    )

    assert report["passed"] is True


def test_benchmark_evidence_gate_cli_returns_nonzero_on_failure(tmp_path: Path) -> None:
    """The command line exits with a failure status when a gate fails."""
    tool = _load_tool()
    manifest = tmp_path / "benchmarks" / "benchmark_regression_gates.json"
    _write_json(
        manifest,
        {"gates": [{"id": "missing", "artefact": "benchmarks/results/missing.json"}]},
    )
    output = tmp_path / "gate.json"

    exit_code = tool.main(["--manifest", str(manifest), "--output", str(output)])

    assert exit_code == 1
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is False


def test_benchmark_evidence_gate_rejects_manifest_without_schema(tmp_path: Path) -> None:
    """A manifest without its schema version is refused."""
    tool = _load_tool()
    manifest = tmp_path / "benchmarks" / "benchmark_regression_gates.json"
    _write_json(manifest, {"gates": []})

    report = tool.evaluate_benchmark_evidence_gate(
        manifest_path=manifest,
        output_path=tmp_path / "gate.json",
        repo_root=tmp_path,
    )

    reasons = {failure["reason"] for failure in report["failures"]}
    assert report["passed"] is False
    assert "manifest_missing_spdx_marker" in reasons
    assert "manifest_schema_version_mismatch" in reasons
    assert "manifest_has_no_gates" in reasons


def test_benchmark_evidence_gate_rejects_duplicate_gate_ids(tmp_path: Path) -> None:
    """Two gates with the same identifier are refused."""
    tool = _load_tool()
    _write_json(tmp_path / "benchmarks" / "results" / "one.json", {"latency_ns": 1.0})
    _write_json(tmp_path / "benchmarks" / "results" / "two.json", {"latency_ns": 2.0})
    manifest = tmp_path / "benchmarks" / "benchmark_regression_gates.json"
    _write_json(
        manifest,
        {
            "SPDX-License-Identifier": "AGPL-3.0-or-later",
            "schema_version": "sc-neurocore.benchmark-regression-gates.v1",
            "gates": [
                {
                    "id": "duplicate",
                    "artefact": "benchmarks/results/one.json",
                    "required_numbers": ["latency_ns"],
                },
                {
                    "id": "duplicate",
                    "artefact": "benchmarks/results/two.json",
                    "required_numbers": ["latency_ns"],
                },
            ],
        },
    )

    report = tool.evaluate_benchmark_evidence_gate(
        manifest_path=manifest,
        output_path=tmp_path / "gate.json",
        repo_root=tmp_path,
    )

    assert report["passed"] is False
    assert any(failure["reason"] == "duplicate_gate_id" for failure in report["failures"])


def test_benchmark_evidence_gate_rejects_contractless_gate(tmp_path: Path) -> None:
    """A gate that requires nothing of its artefact is refused."""
    tool = _load_tool()
    _write_json(tmp_path / "benchmarks" / "results" / "demo.json", {"latency_ns": 1.0})
    manifest = tmp_path / "benchmarks" / "benchmark_regression_gates.json"
    _write_json(
        manifest,
        {
            "SPDX-License-Identifier": "AGPL-3.0-or-later",
            "schema_version": "sc-neurocore.benchmark-regression-gates.v1",
            "gates": [{"id": "weak", "artefact": "benchmarks/results/demo.json"}],
        },
    )

    report = tool.evaluate_benchmark_evidence_gate(
        manifest_path=manifest,
        output_path=tmp_path / "gate.json",
        repo_root=tmp_path,
    )

    assert report["passed"] is False
    assert any(
        failure["reason"] == "gate_has_no_required_metrics_or_contracts"
        for failure in report["failures"]
    )


def test_benchmark_evidence_gate_accepts_numeric_parity_group(tmp_path: Path) -> None:
    """Numbers declared equal across lanes pass when they are equal."""
    tool = _load_tool()
    _write_json(
        tmp_path / "benchmarks" / "results" / "demo.json",
        {
            "backend_summary": {
                "python": {"spikes": 881},
                "rust": {"spikes": 881},
                "go": {"spikes": 881},
            }
        },
    )
    manifest = tmp_path / "benchmarks" / "benchmark_regression_gates.json"
    _write_json(
        manifest,
        {
            "SPDX-License-Identifier": "AGPL-3.0-or-later",
            "schema_version": "sc-neurocore.benchmark-regression-gates.v1",
            "gates": [
                {
                    "id": "parity",
                    "artefact": "benchmarks/results/demo.json",
                    "required_numbers": [
                        "backend_summary.python.spikes",
                        "backend_summary.rust.spikes",
                        "backend_summary.go.spikes",
                    ],
                    "parity_groups": [
                        {
                            "paths": [
                                "backend_summary.python.spikes",
                                "backend_summary.rust.spikes",
                                "backend_summary.go.spikes",
                            ],
                            "tolerance": 0,
                        }
                    ],
                }
            ],
        },
    )

    report = tool.evaluate_benchmark_evidence_gate(
        manifest_path=manifest,
        output_path=tmp_path / "gate.json",
        repo_root=tmp_path,
    )

    assert report["passed"] is True


def test_benchmark_evidence_gate_rejects_numeric_parity_mismatch(tmp_path: Path) -> None:
    """Numbers declared equal across lanes fail when they differ."""
    tool = _load_tool()
    _write_json(
        tmp_path / "benchmarks" / "results" / "demo.json",
        {
            "backend_summary": {
                "python": {"spikes": 881},
                "rust": {"spikes": 881},
                "go": {"spikes": 880},
            }
        },
    )
    manifest = tmp_path / "benchmarks" / "benchmark_regression_gates.json"
    _write_json(
        manifest,
        {
            "SPDX-License-Identifier": "AGPL-3.0-or-later",
            "schema_version": "sc-neurocore.benchmark-regression-gates.v1",
            "gates": [
                {
                    "id": "parity",
                    "artefact": "benchmarks/results/demo.json",
                    "required_numbers": ["backend_summary.python.spikes"],
                    "parity_groups": [
                        {
                            "paths": [
                                "backend_summary.python.spikes",
                                "backend_summary.rust.spikes",
                                "backend_summary.go.spikes",
                            ],
                            "tolerance": 0,
                        }
                    ],
                }
            ],
        },
    )

    report = tool.evaluate_benchmark_evidence_gate(
        manifest_path=manifest,
        output_path=tmp_path / "gate.json",
        repo_root=tmp_path,
    )

    assert report["passed"] is False
    assert any(failure["reason"] == "parity_group_mismatch" for failure in report["failures"])


def test_repository_gate_cli_reports_its_real_verdict(tmp_path: Path) -> None:
    """Exercise the committed manifest through the same CLI used by CI."""
    repo_root = Path(__file__).resolve().parents[2]
    output = tmp_path / "gate-report.json"
    result = subprocess.run(
        [sys.executable, "tools/benchmark_evidence_gate.py", "--output", str(output)],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    manifest = json.loads(
        (repo_root / "benchmarks" / "benchmark_regression_gates.json").read_text(encoding="utf-8")
    )

    assert report["gate_count"] == len(manifest["gates"])
    assert report["source_hash_policy"] == "artefact_commit"
    assert result.returncode == (0 if report["passed"] else 1)

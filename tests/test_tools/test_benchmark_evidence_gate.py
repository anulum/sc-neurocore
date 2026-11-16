# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Hold the benchmark evidence gate to real records, sources and Git history."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
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


_BYTE_NAMES = pytest.mark.skipif(
    sys.platform != "linux", reason="a file name is an arbitrary byte string only on Linux"
)


@_BYTE_NAMES
def test_tracked_files_lists_a_name_that_is_not_utf8(tmp_path: Path) -> None:
    """Such a name is listed in the form a directory walk gives it, so the two can be compared."""
    tool = _load_tool()
    repo = tmp_path / "repo"
    _tracked_and_untracked_tree(repo)
    results = repo / "benchmarks" / "results"
    odd = os.fsencode(results) + b"/record_\xe9.json"
    with open(odd, "wb") as handle:
        handle.write(b"{}")
    subprocess.run([b"git", b"-C", os.fsencode(repo), b"add", odd], check=True, capture_output=True)
    listed = tool.tracked_files(repo, "benchmarks/results")
    assert listed == (
        os.fsdecode(b"benchmarks/results/record_\xe9.json"),
        "benchmarks/results/tracked.json",
    )
    walked = {path.relative_to(repo).as_posix() for path in results.glob("*.json")}
    assert set(listed) < walked


@_BYTE_NAMES
def test_tracked_files_reads_a_work_tree_whose_own_path_is_not_utf8(tmp_path: Path) -> None:
    """The top of the work tree is compared in the same form, so such a tree is still a repository."""
    tool = _load_tool()
    repo = Path(os.fsdecode(os.fsencode(tmp_path) + b"/rep\xe9"))
    _tracked_and_untracked_tree(repo)
    assert tool.tracked_files(repo, "benchmarks/results") == ("benchmarks/results/tracked.json",)


@pytest.mark.skipif(
    sys.platform == "win32", reason="Windows removes a trailing space from a directory name"
)
def test_tracked_files_reads_a_work_tree_whose_name_ends_with_a_space(tmp_path: Path) -> None:
    """Only the line end Git prints is removed from the top it reports, not a space of the name."""
    tool = _load_tool()
    repo = tmp_path / "repo "
    _tracked_and_untracked_tree(repo)
    assert tool.tracked_files(repo, "benchmarks/results") == ("benchmarks/results/tracked.json",)


@_BYTE_NAMES
def test_tracked_files_reads_a_work_tree_whose_name_ends_with_a_line_end(tmp_path: Path) -> None:
    """Git prints one line end after the top; a line end that belongs to the name stays."""
    tool = _load_tool()
    repo = tmp_path / "repo\n"
    _tracked_and_untracked_tree(repo)
    assert tool.tracked_files(repo, "benchmarks/results") == ("benchmarks/results/tracked.json",)


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


#: The tracked gate report the CI step holds the gate's output to.
TRACKED_REPORT = "benchmarks/results/benchmark_evidence_gate_report.json"


def _ci_gate_step() -> dict[str, Any]:
    """Return the CI lint job's step that runs the benchmark evidence gate.

    Returns
    -------
    dict
        The step as the workflow states it.
    """
    import yaml

    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((root / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["lint"]["steps"]
    return next(step for step in steps if step.get("name") == "Historical benchmark evidence")


def _run_ci_gate_step(tmp_path: Path, written: bytes) -> subprocess.CompletedProcess[str]:
    """Run the CI step's own script with a ``python`` that writes the given report.

    The script runs as the workflow runs it, in the repository, with ``RUNNER_TEMP``
    a fresh folder. Only the gate is replaced: a stand-in ``python`` on ``PATH``
    writes ``written`` to the path after ``--output``, so the comparison that
    follows is the step's own.

    Parameters
    ----------
    tmp_path : Path
        A fresh folder for the stand-in and for ``RUNNER_TEMP``.
    written : bytes
        What the stand-in gate writes as its report.

    Returns
    -------
    subprocess.CompletedProcess
        The script's exit status and output.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    payload = tmp_path / "payload.json"
    payload.write_bytes(written)
    stand_in = bin_dir / "python"
    stand_in.write_text(
        "#!/bin/sh\n"
        'while [ "$#" -gt 0 ]; do\n'
        '  if [ "$1" = "--output" ]; then cp "' + str(payload) + '" "$2"; exit 0; fi\n'
        "  shift\n"
        "done\n"
        "exit 2\n",
        encoding="utf-8",
    )
    stand_in.chmod(0o755)
    runner_temp = tmp_path / "runner"
    runner_temp.mkdir()
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    env["RUNNER_TEMP"] = str(runner_temp)
    return subprocess.run(
        ["bash", "-e", "-c", _ci_gate_step()["run"]],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_ci_holds_the_gate_report_to_the_tracked_one() -> None:
    """CI writes the gate's report to its own temporary folder and compares it with the tracked one.

    The report holds gate ids, paths, failures, counts, the verdict and the
    policy, so a change that moves any of them must commit the report again.
    """
    run = _ci_gate_step()["run"]
    assert (
        'tools/benchmark_evidence_gate.py --output "$RUNNER_TEMP/benchmark-evidence-gate.json"'
        in run
    )
    assert f'cmp -s "$RUNNER_TEMP/benchmark-evidence-gate.json" {TRACKED_REPORT}' in run


def test_the_ci_step_fails_on_a_report_that_differs(tmp_path: Path) -> None:
    """A gate output other than the tracked report fails the step and names the file."""
    result = _run_ci_gate_step(tmp_path, b'{"passed": true}\n')
    assert result.returncode == 1
    assert (
        f"::error file={TRACKED_REPORT}::the committed benchmark evidence gate report is not "
        "what the gate writes; run python tools/benchmark_evidence_gate.py and commit "
        f"{TRACKED_REPORT}"
    ) in result.stdout
    assert f"--- {TRACKED_REPORT}" in result.stdout


def test_the_ci_step_passes_when_the_report_is_the_tracked_one(tmp_path: Path) -> None:
    """The tracked report's own bytes pass the comparison."""
    root = Path(__file__).resolve().parents[2]
    result = _run_ci_gate_step(tmp_path, (root / TRACKED_REPORT).read_bytes())
    assert result.returncode == 0, result.stdout + result.stderr
    assert "::error" not in result.stdout


#: Records that key their source digests by runtime lane, each with the
#: benchmark script that wrote it and the field that holds the digests.
LANE_KEYED_RECORDS = {
    "benchmarks/results/bench_bertram_phantom.json": (
        "benchmarks/bench_model_bertram_phantom.py",
        "source_hashes",
    ),
    "benchmarks/results/bench_glif_simulate.json": (
        "benchmarks/bench_glif_simulate.py",
        "source_sha256",
    ),
    "benchmarks/results/bench_sc_four_state_glif.json": (
        "benchmarks/bench_sc_four_state_glif.py",
        "source_sha256",
    ),
}


def _script_lane_paths(script: Path) -> dict[str, str]:
    """Read the lane-to-path map a benchmark script assigns to ``sources``.

    Parameters
    ----------
    script : Path
        A benchmark script whose ``sources`` map is a literal: each value a
        repository path, as a string or as ``ROOT / "path"``.

    Returns
    -------
    dict
        Lane name to repository path.
    """
    tree = ast.parse(script.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "sources" for target in node.targets
        ):
            continue
        lanes: dict[str, str] = {}
        for key, value in zip(node.value.keys, node.value.values, strict=True):
            if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Div):
                value = value.right
            assert isinstance(key, ast.Constant) and isinstance(value, ast.Constant)
            lanes[str(key.value)] = str(value.value)
        return lanes
    raise AssertionError(f"{script} assigns no literal sources map")


@pytest.mark.parametrize("artefact", sorted(LANE_KEYED_RECORDS))
def test_a_lane_keyed_record_is_bound_by_the_paths_its_script_hashed(artefact: str) -> None:
    """The manifest maps each lane digest to the file the script hashed for that lane.

    Such a record names no path itself, so the gate's map is what binds it to
    its sources: the gate checks the digests at the record's commit, and the
    benchmark coverage census reads the same map for the sources now.
    """
    root = Path(__file__).resolve().parents[2]
    script, field = LANE_KEYED_RECORDS[artefact]
    manifest = json.loads(
        (root / "benchmarks" / "benchmark_regression_gates.json").read_text(encoding="utf-8")
    )
    gates = [gate for gate in manifest["gates"] if gate["artefact"] == artefact]
    assert len(gates) == 1
    lanes = _script_lane_paths(root / script)
    assert gates[0]["source_hashes"] == {path: f"{field}.{lane}" for lane, path in lanes.items()}
    record = json.loads((root / artefact).read_text(encoding="utf-8"))
    assert set(record[field]) == set(lanes)


#: The MNIST accuracy record, which names its checkpoint and training log by
#: path beside their digests.
MNIST_RECORD = "benchmarks/results/mnist_conv_accuracy_reproducibility.json"


def test_the_mnist_record_is_bound_by_the_files_it_names() -> None:
    """The manifest maps the checkpoint's and the log's digests to the paths the record gives."""
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads(
        (root / "benchmarks" / "benchmark_regression_gates.json").read_text(encoding="utf-8")
    )
    gates = [gate for gate in manifest["gates"] if gate["artefact"] == MNIST_RECORD]
    assert len(gates) == 1
    record = json.loads((root / MNIST_RECORD).read_text(encoding="utf-8"))
    assert gates[0]["source_hashes"] == {
        record["checkpoint"]["path"]: "checkpoint.sha256",
        record["training_log"]["path"]: "training_log.sha256",
    }


def test_the_tracked_report_passes_the_gates_that_map_digests() -> None:
    """The tracked report evaluates the four gates and holds no failure of theirs."""
    root = Path(__file__).resolve().parents[2]
    report = json.loads((root / TRACKED_REPORT).read_text(encoding="utf-8"))
    manifest = json.loads(
        (root / "benchmarks" / "benchmark_regression_gates.json").read_text(encoding="utf-8")
    )
    artefacts = {*LANE_KEYED_RECORDS, MNIST_RECORD}
    ids = {gate["id"] for gate in manifest["gates"] if gate["artefact"] in artefacts}
    evaluated = {entry["id"] for entry in report["evaluated_gates"]}
    assert len(ids) == len(artefacts)
    assert ids <= evaluated
    assert not [failure for failure in report["failures"] if failure["gate_id"] in ids]


#: Gates whose record also hashes sources inside a list, once per variant, as ``variants.<i>.source_hashes.<path>``.
#: The gate's ``_path_value`` stops at a list, and a map keyed by source path points at one place per path, so these
#: digests are not checked against the source. Their number per gate is held here, so a new one is noticed.
_LIST_HELD_DIGESTS = {
    "autonomous-learning-modularisation-local-diagnostics": 32,
    "bioware-modularisation-local-diagnostics": 14,
    "nir-graph-modularisation-local-diagnostics": 26,
}


def _is_digest(value: Any) -> bool:
    """Return whether ``value`` is a lowercase hexadecimal SHA-256 digest.

    Parameters
    ----------
    value : Any
        A value read from a record.
    """
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _path_keyed_digests(
    record: Any, suffixes: tuple[str, ...], prefix: str = ""
) -> dict[str, tuple[str, str]]:
    """Return every place in a record's tables, at any depth, that holds the SHA-256 of a repository path.

    Parameters
    ----------
    record : Any
        A committed benchmark record, or a table within one.
    suffixes : tuple of str
        The file suffixes that mark a key as a repository path, as the gate reads them.
    prefix : str
        The dotted place of ``record`` in the whole record; empty for the record itself.

    Returns
    -------
    dict of str to tuple of (str, str)
        Each dotted place, as the gate's ``_path_value`` reads it, with the path and the digest held there. A list is
        not entered; ``_list_held_digests`` counts what lies inside one.
    """
    places: dict[str, tuple[str, str]] = {}
    if not isinstance(record, dict):
        return places
    for key, value in record.items():
        place = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            places.update(_path_keyed_digests(value, suffixes, place))
        elif key.endswith(suffixes) and _is_digest(value):
            places[place] = (key, value)
    return places


def _list_held_digests(value: Any, suffixes: tuple[str, ...], in_list: bool = False) -> int:
    """Count the path-keyed digests that sit somewhere inside a list of a record.

    Parameters
    ----------
    value : Any
        A committed benchmark record, or a part of one.
    suffixes : tuple of str
        The file suffixes that mark a key as a repository path, as the gate reads them.
    in_list : bool
        Whether ``value`` lies inside a list.

    Returns
    -------
    int
        The number of such digests.
    """
    if isinstance(value, list):
        return sum(_list_held_digests(item, suffixes, True) for item in value)
    if not isinstance(value, dict):
        return 0
    return sum(
        1
        if in_list and key.endswith(suffixes) and _is_digest(item)
        else _list_held_digests(item, suffixes, in_list)
        for key, item in value.items()
    )


def _unchecked_digests(gate: dict[str, Any], record: Any, tool: Any) -> list[str]:
    """Return the places of a record's path-keyed table digests that its gate does not check.

    A digest is checked when the gate's map points at its place. It is also checked when the map points at another
    place for the same path that holds the same digest, since the gate compares that one with the source.

    Parameters
    ----------
    gate : dict of str to Any
        A manifest gate with a ``source_hashes`` map.
    record : Any
        The gate's committed record.
    tool : Any
        The loaded gate module.

    Returns
    -------
    list of str
        The unchecked places, sorted.
    """
    mapping = gate["source_hashes"]
    mapped = set(mapping.values())
    unchecked = []
    for place, (path, digest) in sorted(
        _path_keyed_digests(record, tool._SOURCE_PATH_SUFFIXES).items()
    ):
        if place in mapped:
            continue
        if path in mapping and tool._path_value(record, mapping[path]) == digest:
            continue
        unchecked.append(place)
    return unchecked


def test_a_digest_is_found_in_a_table_at_any_depth_but_not_in_a_list() -> None:
    """The walk reads tables at any depth, each place one the gate can address, and leaves lists to their count."""
    tool = _load_tool()
    digest = "ab" * 32
    record = {
        "src/top.py": digest,
        "source_hashes": {"src/a.py": digest, "label": "x", "src/short.py": "abc"},
        "source_manifest": {"files": {"engine/src/b.rs": digest}, "runner_sha256": digest},
        "variants": [{"source_hashes": {"src/c.py": digest}}, {"name": "v2"}],
    }
    places = _path_keyed_digests(record, (".py", ".rs"))
    assert places == {
        "src/top.py": ("src/top.py", digest),
        "source_hashes.src/a.py": ("src/a.py", digest),
        "source_manifest.files.engine/src/b.rs": ("engine/src/b.rs", digest),
    }
    assert {place: tool._path_value(record, place) for place in places} == {
        place: held for place, (_path, held) in places.items()
    }
    assert _list_held_digests(record, (".py", ".rs")) == 1


def test_a_digest_held_twice_is_checked_through_its_mapped_twin_only_when_equal() -> None:
    """A second digest of a mapped path is checked only when it equals the one the gate compares."""
    tool = _load_tool()
    digest, other = "ab" * 32, "cd" * 32
    gate = {"source_hashes": {"src/a.py": "gate_hashes.src__a_py"}}
    twin = {"gate_hashes": {"src__a_py": digest}, "manifest": {"files": {"src/a.py": digest}}}
    differs = {"gate_hashes": {"src__a_py": digest}, "manifest": {"files": {"src/a.py": other}}}
    unmapped = {"gate_hashes": {"src__a_py": digest}, "manifest": {"files": {"src/b.py": digest}}}
    assert _unchecked_digests(gate, twin, tool) == []
    assert _unchecked_digests(gate, differs, tool) == ["manifest.files.src/a.py"]
    assert _unchecked_digests(gate, unmapped, tool) == ["manifest.files.src/b.py"]


def _mapped_gates_and_records() -> list[tuple[dict[str, Any], Any]]:
    """Return each manifest gate with a ``source_hashes`` map, with its committed record."""
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads(
        (root / "benchmarks" / "benchmark_regression_gates.json").read_text(encoding="utf-8")
    )
    return [
        (gate, json.loads((root / gate["artefact"]).read_text(encoding="utf-8")))
        for gate in manifest["gates"]
        if isinstance(gate.get("source_hashes"), dict)
    ]


def test_every_manifest_map_checks_every_path_digest_in_its_record_tables() -> None:
    """A gate that maps its record's digests checks every one its record's tables hold, at any depth.

    A source the record hashes but the gate does not check could change, or differ from the record's own digest at
    its commit, and the gate would not say so.
    """
    tool = _load_tool()
    pairs = _mapped_gates_and_records()
    assert pairs
    gaps = {gate["id"]: _unchecked_digests(gate, record, tool) for gate, record in pairs}
    assert {gate_id: places for gate_id, places in gaps.items() if places} == {}


def test_the_digests_held_in_lists_are_the_known_ones() -> None:
    """Only the known records hash sources inside a list, each with its known number of digests."""
    tool = _load_tool()
    held = {
        gate["id"]: _list_held_digests(record, tool._SOURCE_PATH_SUFFIXES)
        for gate, record in _mapped_gates_and_records()
    }
    assert {gate_id: count for gate_id, count in held.items() if count} == _LIST_HELD_DIGESTS

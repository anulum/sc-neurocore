# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — engine crate-root provenance gate tests

"""Hermetic fixtures for the engine crate-root benchmark-evidence provenance gate."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

CRATE_ROOT_BODY = b"pub mod neurons;\npub mod engine_v2;\n"
ENGINE = "engine/src/lib.rs"


def _load_tool() -> Any:
    """Load the crate-root provenance gate module by path (tools/ is not a package)."""
    repo_root = Path(__file__).resolve().parents[2]
    tool_path = repo_root / "tools" / "lib_rs_provenance_gate.py"
    spec = importlib.util.spec_from_file_location("lib_rs_provenance_gate", tool_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


def _crate_root_sha() -> str:
    """Return the sha256 of the fixture crate-root body."""
    return hashlib.sha256(CRATE_ROOT_BODY).hexdigest()


def _write(path: Path, payload: object) -> None:
    """Write a JSON payload, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Return a tmp repository root with a fixture engine crate root in place."""
    crate = tmp_path / "engine" / "src" / "lib.rs"
    crate.parent.mkdir(parents=True, exist_ok=True)
    crate.write_bytes(CRATE_ROOT_BODY)
    (tmp_path / "benchmarks" / "results").mkdir(parents=True, exist_ok=True)
    return tmp_path


def test_evaluate_passes_when_every_binding_is_fresh(repo: Path) -> None:
    """Accept a tree whose committed crate-root hashes all match the live crate root."""
    live = _crate_root_sha()
    _write(repo / "benchmarks/results/bench_a.json", {"source_hashes": {ENGINE: live}})
    _write(
        repo / "benchmarks/results/bench_b.json",
        {"source_sha256": {ENGINE: live, "x.py": "0" * 64}},
    )
    _write(
        repo / "benchmarks/results/bench_no_binding.json", {"source_hashes": {"only.py": "1" * 64}}
    )
    _write(repo / "benchmarks/results/bench_aggregate.json", {"source_sha256": "f" * 64})
    assert tool.evaluate(repo_root=repo) == []


def test_evaluate_flags_stale_binding(repo: Path) -> None:
    """Reject a committed crate-root hash that disagrees with the live crate root."""
    _write(
        repo / "benchmarks/results/bench_stale.json",
        {"source_hashes": {"engine/src/lib.rs": "0" * 64}},
    )
    failures = tool.evaluate(repo_root=repo)
    assert len(failures) == 1
    failure = failures[0]
    assert failure.artefact == "benchmarks/results/bench_stale.json"
    assert failure.reason == "stale_engine_crate_root_hash"
    assert failure.recorded == "0" * 64
    assert failure.expected == _crate_root_sha()
    assert failure.to_json() == {
        "artefact": "benchmarks/results/bench_stale.json",
        "recorded": "0" * 64,
        "expected": _crate_root_sha(),
        "reason": "stale_engine_crate_root_hash",
    }


def test_evaluate_flags_stale_alternate_convention(repo: Path) -> None:
    """Reject a stale crate-root hash recorded under the ``source_sha256`` convention."""
    _write(
        repo / "benchmarks/results/bench_alt.json",
        {"source_sha256": {"engine/src/lib.rs": "a" * 64}},
    )
    failures = tool.evaluate(repo_root=repo)
    assert [f.artefact for f in failures] == ["benchmarks/results/bench_alt.json"]


def test_evaluate_ignores_nested_variant_blocks(repo: Path) -> None:
    """Skip historical ``variants[*]`` provenance blocks; gate only the canonical map."""
    _write(
        repo / "benchmarks/results/bench_variant.json",
        {
            "source_hashes": {"engine/src/lib.rs": _crate_root_sha()},
            "variants": [{"source_hashes": {"engine/src/lib.rs": "9" * 64}}],
        },
    )
    assert tool.evaluate(repo_root=repo) == []


def test_evaluate_rejects_non_sha_and_non_dict_maps(repo: Path) -> None:
    """Reject malformed provenance maps and crate-root digests instead of passing open."""
    _write(repo / "benchmarks/results/bench_bad_map.json", {"source_hashes": 123})
    _write(
        repo / "benchmarks/results/bench_short.json",
        {"source_hashes": {"engine/src/lib.rs": "short"}},
    )
    failures = tool.evaluate(repo_root=repo)
    assert {failure.reason for failure in failures} == {
        "invalid_engine_crate_root_hash",
        "invalid_provenance_map",
    }


def test_evaluate_rejects_malformed_but_allows_non_object_json(repo: Path) -> None:
    """Reject malformed JSON while allowing legitimate benchmark-array artefacts."""
    (repo / "benchmarks/results/broken.json").write_text("{ not json", encoding="utf-8")
    (repo / "benchmarks/results/list.json").write_text("[]", encoding="utf-8")
    failures = tool.evaluate(repo_root=repo)
    assert len(failures) == 1
    assert failures[0].artefact == "benchmarks/results/broken.json"
    assert failures[0].reason == "invalid_benchmark_json"


def test_evaluate_reports_missing_crate_root(tmp_path: Path) -> None:
    """Fail closed when the engine crate root is absent from the tree."""
    (tmp_path / "benchmarks").mkdir()
    failures = tool.evaluate(repo_root=tmp_path)
    assert len(failures) == 1
    assert failures[0].reason == "engine_crate_root_missing"


def test_evaluate_handles_benchmarks_root_outside_repo(tmp_path: Path) -> None:
    """Report an absolute artefact path when the benchmarks tree is outside the repo root."""
    repo_root = tmp_path / "repo"
    crate = repo_root / "engine" / "src" / "lib.rs"
    crate.parent.mkdir(parents=True, exist_ok=True)
    crate.write_bytes(CRATE_ROOT_BODY)
    external = tmp_path / "outside"
    external.mkdir()
    _write(external / "bench_ext.json", {"source_hashes": {"engine/src/lib.rs": "0" * 64}})
    failures = tool.evaluate(repo_root=repo_root, benchmarks_root=external)
    assert len(failures) == 1
    assert failures[0].artefact == (external / "bench_ext.json").as_posix()


def test_iter_yields_both_conventions(repo: Path) -> None:
    """Enumerate crate-root bindings recorded under either supported convention."""
    live = _crate_root_sha()
    _write(repo / "benchmarks/results/one.json", {"source_hashes": {"engine/src/lib.rs": live}})
    _write(repo / "benchmarks/results/two.json", {"source_sha256": {"engine/src/lib.rs": live}})
    found = {path.name for path, _ in tool.iter_crate_root_bindings(repo / "benchmarks")}
    assert found == {"one.json", "two.json"}


def test_cli_returns_zero_when_clean(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Return zero and emit a passing report when no crate-root binding is stale."""
    _write(
        repo / "benchmarks/results/bench_ok.json",
        {"source_hashes": {"engine/src/lib.rs": _crate_root_sha()}},
    )
    assert tool.main(["--repo-root", str(repo)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is True
    assert report["failure_count"] == 0


def test_cli_returns_nonzero_on_stale(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Return one and emit the stale artefact when a crate-root binding is stale."""
    _write(
        repo / "benchmarks/results/bench_bad.json",
        {"source_hashes": {"engine/src/lib.rs": "0" * 64}},
    )
    assert tool.main(["--repo-root", str(repo), "--benchmarks-root", str(repo / "benchmarks")]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is False
    assert report["failures"][0]["artefact"] == "benchmarks/results/bench_bad.json"

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for the engine crate-root monolith no-growth guard

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_tool() -> Any:
    tool_path = _repo_root() / "tools" / "engine_monolith_guard.py"
    spec = importlib.util.spec_from_file_location("engine_monolith_guard", tool_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_fake_crate(root: Path, *, lines: int, pyfunctions: int) -> None:
    """Write ``engine/src/lib.rs`` with an exact line and pyfunction count."""

    assert lines >= 2 * pyfunctions, "need two lines per pyfunction (attr + fn)"
    body: list[str] = []
    for index in range(pyfunctions):
        body.append("#[pyfunction]")
        body.append(f"fn f{index}() {{}}")
    while len(body) < lines:
        body.append("// pad")
    src = root / "engine" / "src"
    src.mkdir(parents=True, exist_ok=True)
    (src / "lib.rs").write_text("\n".join(body[:lines]) + "\n", encoding="utf-8")


def _write_ceiling(path: Path, rel: str, *, max_lines: int, max_pyfunctions: int) -> None:
    path.write_text(
        "schema_version = 1\n"
        f'[targets."{rel}"]\n'
        f"max_lines = {max_lines}\n"
        f"max_pyfunctions = {max_pyfunctions}\n",
        encoding="utf-8",
    )


def test_pyfunction_pattern_matches_plain_and_arg_forms() -> None:
    tool = _load_tool()
    text = "#[pyfunction]\n#[pyfunction(signature = (x))]\n#[pyfunctions]\nfn pyfunction_x() {}"
    matches = tool.PYFUNCTION_PATTERN.findall(text)
    # Plain and argument forms count; the plural identifier and a bare fn name do not.
    assert len(matches) == 2


def test_measure_target_matches_independent_recount() -> None:
    tool = _load_tool()
    repo = _repo_root()
    actual = tool.measure_target(repo, "engine/src/lib.rs")
    text = (repo / "engine" / "src" / "lib.rs").read_text(encoding="utf-8")
    assert actual["lines"] == len(text.splitlines())
    assert actual["pyfunctions"] == text.count("#[pyfunction]") + text.count("#[pyfunction(")
    assert actual["lines"] > 0
    assert actual["pyfunctions"] > 0


def test_live_tree_is_within_committed_ceiling() -> None:
    tool = _load_tool()
    repo = _repo_root()
    ceiling = tool.load_ceiling(repo / tool.DEFAULT_CEILING)
    report = tool.evaluate(repo, ceiling)
    assert report["passed"], report["violations"]


def test_growth_over_ceiling_is_flagged(tmp_path: Path) -> None:
    tool = _load_tool()
    _write_fake_crate(tmp_path, lines=40, pyfunctions=10)
    ceiling = {
        "schema_version": 1,
        "targets": {"engine/src/lib.rs": {"max_lines": 30, "max_pyfunctions": 8}},
    }
    report = tool.evaluate(tmp_path, ceiling)
    assert report["passed"] is False
    by_metric = {v["metric"]: v for v in report["violations"]}
    assert by_metric["lines"]["actual"] == 40
    assert by_metric["lines"]["delta"] == 10
    assert by_metric["pyfunctions"]["actual"] == 10
    assert by_metric["pyfunctions"]["delta"] == 2


def test_load_ceiling_rejects_bad_schema(tmp_path: Path) -> None:
    tool = _load_tool()
    path = tmp_path / "ceiling.toml"
    path.write_text('schema_version = 99\n[targets."a"]\nmax_lines = 1\n', encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        tool.load_ceiling(path)


def test_load_ceiling_rejects_missing_targets(tmp_path: Path) -> None:
    tool = _load_tool()
    path = tmp_path / "ceiling.toml"
    path.write_text("schema_version = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no \\[targets\\]"):
        tool.load_ceiling(path)


def test_tightened_ceiling_lowers_after_reduction(tmp_path: Path) -> None:
    tool = _load_tool()
    _write_fake_crate(tmp_path, lines=20, pyfunctions=5)
    ceiling = {
        "schema_version": 1,
        "targets": {"engine/src/lib.rs": {"max_lines": 30, "max_pyfunctions": 8}},
    }
    tightened = tool.tightened_ceiling(tmp_path, ceiling)
    assert tightened["targets"]["engine/src/lib.rs"] == {"max_lines": 20, "max_pyfunctions": 5}


def test_tightened_ceiling_refuses_to_raise(tmp_path: Path) -> None:
    tool = _load_tool()
    _write_fake_crate(tmp_path, lines=40, pyfunctions=10)
    ceiling = {
        "schema_version": 1,
        "targets": {"engine/src/lib.rs": {"max_lines": 30, "max_pyfunctions": 8}},
    }
    with pytest.raises(tool.CeilingRaiseError, match="refusing to raise"):
        tool.tightened_ceiling(tmp_path, ceiling)


def test_render_toml_roundtrips(tmp_path: Path) -> None:
    tool = _load_tool()
    ceiling = {
        "schema_version": 1,
        "targets": {"engine/src/lib.rs": {"max_lines": 7116, "max_pyfunctions": 212}},
    }
    rendered = tmp_path / "out.toml"
    rendered.write_text(tool.render_ceiling_toml(ceiling), encoding="utf-8")
    reloaded = tool.load_ceiling(rendered)
    assert reloaded["targets"] == ceiling["targets"]
    assert "SPDX-License-Identifier" in rendered.read_text(encoding="utf-8")


def test_committed_ceiling_is_canonical() -> None:
    tool = _load_tool()
    path = _repo_root() / tool.DEFAULT_CEILING
    ceiling = tool.load_ceiling(path)
    assert tool.render_ceiling_toml(ceiling) == path.read_text(encoding="utf-8")


def test_main_check_passes_on_live_tree(capsys: pytest.CaptureFixture[str]) -> None:
    tool = _load_tool()
    assert tool.main(["--check", "--repo", str(_repo_root())]) == 0
    assert "engine/src/lib.rs" in capsys.readouterr().out


def test_main_check_fails_on_growth(tmp_path: Path) -> None:
    tool = _load_tool()
    _write_fake_crate(tmp_path, lines=40, pyfunctions=10)
    ceiling = tmp_path / "ceiling.toml"
    _write_ceiling(ceiling, "engine/src/lib.rs", max_lines=30, max_pyfunctions=8)
    assert tool.main(["--check", "--repo", str(tmp_path), "--ceiling", str(ceiling)]) == 1


def test_main_update_writes_lowered_ceiling(tmp_path: Path) -> None:
    tool = _load_tool()
    _write_fake_crate(tmp_path, lines=20, pyfunctions=5)
    ceiling = tmp_path / "ceiling.toml"
    _write_ceiling(ceiling, "engine/src/lib.rs", max_lines=30, max_pyfunctions=8)
    assert tool.main(["--update", "--repo", str(tmp_path), "--ceiling", str(ceiling)]) == 0
    reloaded = tool.load_ceiling(ceiling)
    assert reloaded["targets"]["engine/src/lib.rs"] == {"max_lines": 20, "max_pyfunctions": 5}


def test_main_update_refuses_raise_returns_one(tmp_path: Path) -> None:
    tool = _load_tool()
    _write_fake_crate(tmp_path, lines=40, pyfunctions=10)
    ceiling = tmp_path / "ceiling.toml"
    _write_ceiling(ceiling, "engine/src/lib.rs", max_lines=30, max_pyfunctions=8)
    before = ceiling.read_text(encoding="utf-8")
    assert tool.main(["--update", "--repo", str(tmp_path), "--ceiling", str(ceiling)]) == 1
    assert ceiling.read_text(encoding="utf-8") == before


def test_main_default_prints_report(capsys: pytest.CaptureFixture[str]) -> None:
    tool = _load_tool()
    assert tool.main(["--repo", str(_repo_root())]) == 0
    assert "#[pyfunction]" in capsys.readouterr().out

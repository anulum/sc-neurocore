# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (evaluate_and_ceiling_schema) from former test_engine_monolith_guard.py

from __future__ import annotations

from engine_monolith_guard_support import *  # noqa: F403

def test_evaluate_checks_every_configured_target(tmp_path: Path) -> None:
    tool = _load_tool()
    _write_fake_target(tmp_path, lines=20, pyfunctions=2)
    _write_fake_target(
        tmp_path,
        rel_path="engine/src/neurons/misc.rs",
        lines=12,
        pyfunctions=0,
    )
    ceiling: dict[str, Any] = {
        "schema_version": 1,
        "targets": {
            "engine/src/lib.rs": {"max_lines": 20, "max_pyfunctions": 2},
            "engine/src/neurons/misc.rs": {"max_lines": 11, "max_pyfunctions": 0},
        },
    }
    report = tool.evaluate(tmp_path, ceiling)
    assert set(report["measurements"]) == set(ceiling["targets"])
    assert report["violations"] == [
        {
            "path": "engine/src/neurons/misc.rs",
            "metric": "lines",
            "ceiling": 11,
            "actual": 12,
            "delta": 1,
        }
    ]


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
    _write_fake_target(tmp_path, lines=20, pyfunctions=5)
    ceiling = {
        "schema_version": 1,
        "targets": {"engine/src/lib.rs": {"max_lines": 30, "max_pyfunctions": 8}},
    }
    tightened = tool.tightened_ceiling(tmp_path, ceiling)
    assert tightened["targets"]["engine/src/lib.rs"] == {"max_lines": 20, "max_pyfunctions": 5}


def test_tightened_ceiling_refuses_to_raise(tmp_path: Path) -> None:
    tool = _load_tool()
    _write_fake_target(tmp_path, lines=40, pyfunctions=10)
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

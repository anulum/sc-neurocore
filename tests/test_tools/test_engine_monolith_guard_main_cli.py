# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (main_cli) from former test_engine_monolith_guard.py

from __future__ import annotations

from engine_monolith_guard_support import *  # noqa: F403

def test_main_check_passes_on_live_tree(capsys: pytest.CaptureFixture[str]) -> None:
    tool = _load_tool()
    assert tool.main(["--check", "--repo", str(_repo_root())]) == 0
    output = capsys.readouterr().out
    assert "engine/src/lib.rs" in output
    assert "engine/src/ir/qformat.rs" in output
    assert "engine/src/ir/qformat/block_floating.rs" in output
    assert "engine/src/ir/qformat/dense_result.rs" in output
    assert "engine/src/ir/qformat/fixed_format.rs" in output
    assert "engine/src/ir/qformat/mixed_dense.rs" in output
    assert "engine/src/neurons/ai_optimized.rs" in output
    assert "engine/src/neurons/biophysical.rs" in output
    assert "engine/src/neurons/cerebellar.rs" in output
    assert "engine/src/neurons/channels.rs" in output
    assert "engine/src/neurons/interneurons.rs" in output
    assert "engine/src/neurons/misc.rs" in output
    assert "engine/src/neurons/motor.rs" in output
    assert "engine/src/neurons/multi_compartment.rs" in output
    assert "engine/src/neurons/rate.rs" in output
    assert "engine/src/neurons/sensory.rs" in output
    assert "engine/src/neurons/trivial.rs" in output
    assert "engine/src/neurons/simple_spiking.rs" in output
    assert "engine/src/pyo3_neurons.rs" in output


def test_script_entrypoint_exits_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    path = _repo_root() / "tools" / "engine_monolith_guard.py"
    monkeypatch.setattr(
        sys,
        "argv",
        [str(path), "--check", "--repo", str(_repo_root())],
    )
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(path), run_name="__main__")
    assert exc_info.value.code == 0


def test_main_check_fails_on_growth(tmp_path: Path) -> None:
    tool = _load_tool()
    _write_fake_target(tmp_path, lines=40, pyfunctions=10)
    ceiling = tmp_path / "ceiling.toml"
    _write_ceiling(ceiling, "engine/src/lib.rs", max_lines=30, max_pyfunctions=8)
    assert tool.main(["--check", "--repo", str(tmp_path), "--ceiling", str(ceiling)]) == 1


def test_main_update_writes_lowered_ceiling(tmp_path: Path) -> None:
    tool = _load_tool()
    _write_fake_target(tmp_path, lines=20, pyfunctions=5)
    ceiling = tmp_path / "ceiling.toml"
    _write_ceiling(ceiling, "engine/src/lib.rs", max_lines=30, max_pyfunctions=8)
    assert tool.main(["--update", "--repo", str(tmp_path), "--ceiling", str(ceiling)]) == 0
    reloaded = tool.load_ceiling(ceiling)
    assert reloaded["targets"]["engine/src/lib.rs"] == {"max_lines": 20, "max_pyfunctions": 5}


def test_main_update_refuses_raise_returns_one(tmp_path: Path) -> None:
    tool = _load_tool()
    _write_fake_target(tmp_path, lines=40, pyfunctions=10)
    ceiling = tmp_path / "ceiling.toml"
    _write_ceiling(ceiling, "engine/src/lib.rs", max_lines=30, max_pyfunctions=8)
    before = ceiling.read_text(encoding="utf-8")
    assert tool.main(["--update", "--repo", str(tmp_path), "--ceiling", str(ceiling)]) == 1
    assert ceiling.read_text(encoding="utf-8") == before


def test_main_default_prints_report(capsys: pytest.CaptureFixture[str]) -> None:
    tool = _load_tool()
    assert tool.main(["--repo", str(_repo_root())]) == 0
    assert "#[pyfunction]" in capsys.readouterr().out

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (measure_and_facade_ceilings) from former test_engine_monolith_guard.py

from __future__ import annotations

from engine_monolith_guard_support import *  # noqa: F403

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
    assert actual["pyfunctions"] == 0


def test_live_tree_is_within_committed_ceiling() -> None:
    tool = _load_tool()
    repo = _repo_root()
    ceiling = tool.load_ceiling(repo / tool.DEFAULT_CEILING)
    report = tool.evaluate(repo, ceiling)
    assert report["passed"], report["violations"]


def test_pyo3_neuron_ceiling_matches_current_down_only_surface() -> None:
    tool = _load_tool()
    repo = _repo_root()
    ceiling = tool.load_ceiling(repo / tool.DEFAULT_CEILING)
    target = ceiling["targets"]["engine/src/pyo3_neurons.rs"]
    assert target == {
        "max_lines": tool.measure_target(repo, "engine/src/pyo3_neurons.rs")["lines"],
        "max_pyfunctions": 0,
    }


def test_qformat_facade_ceiling_matches_current_down_only_surface() -> None:
    tool = _load_tool()
    repo = _repo_root()
    ceiling = tool.load_ceiling(repo / tool.DEFAULT_CEILING)
    target = ceiling["targets"]["engine/src/ir/qformat.rs"]
    assert target == {
        "max_lines": tool.measure_target(repo, "engine/src/ir/qformat.rs")["lines"],
        "max_pyfunctions": 0,
    }


def test_ai_optimized_facade_ceiling_matches_current_down_only_surface() -> None:
    tool = _load_tool()
    repo = _repo_root()
    ceiling = tool.load_ceiling(repo / tool.DEFAULT_CEILING)
    target = ceiling["targets"]["engine/src/neurons/ai_optimized.rs"]
    assert target == {
        "max_lines": tool.measure_target(repo, "engine/src/neurons/ai_optimized.rs")["lines"],
        "max_pyfunctions": 0,
    }


def test_channels_facade_ceiling_matches_current_down_only_surface() -> None:
    tool = _load_tool()
    repo = _repo_root()
    ceiling = tool.load_ceiling(repo / tool.DEFAULT_CEILING)
    target = ceiling["targets"]["engine/src/neurons/channels.rs"]
    assert target == {
        "max_lines": tool.measure_target(repo, "engine/src/neurons/channels.rs")["lines"],
        "max_pyfunctions": 0,
    }


def test_interneurons_facade_ceiling_matches_current_down_only_surface() -> None:
    tool = _load_tool()
    repo = _repo_root()
    ceiling = tool.load_ceiling(repo / tool.DEFAULT_CEILING)
    target = ceiling["targets"]["engine/src/neurons/interneurons.rs"]
    assert target == {
        "max_lines": tool.measure_target(repo, "engine/src/neurons/interneurons.rs")["lines"],
        "max_pyfunctions": 0,
    }


def test_motor_facade_ceiling_matches_current_down_only_surface() -> None:
    tool = _load_tool()
    repo = _repo_root()
    ceiling = tool.load_ceiling(repo / tool.DEFAULT_CEILING)
    target = ceiling["targets"]["engine/src/neurons/motor.rs"]
    assert target == {
        "max_lines": tool.measure_target(repo, "engine/src/neurons/motor.rs")["lines"],
        "max_pyfunctions": 0,
    }


def test_multi_compartment_facade_ceiling_matches_current_down_only_surface() -> None:
    tool = _load_tool()
    repo = _repo_root()
    ceiling = tool.load_ceiling(repo / tool.DEFAULT_CEILING)
    target = ceiling["targets"]["engine/src/neurons/multi_compartment.rs"]
    assert target == {
        "max_lines": tool.measure_target(repo, "engine/src/neurons/multi_compartment.rs")["lines"],
        "max_pyfunctions": 0,
    }


def test_rate_facade_ceiling_matches_current_down_only_surface() -> None:
    tool = _load_tool()
    repo = _repo_root()
    ceiling = tool.load_ceiling(repo / tool.DEFAULT_CEILING)
    target = ceiling["targets"]["engine/src/neurons/rate.rs"]
    assert target == {
        "max_lines": tool.measure_target(repo, "engine/src/neurons/rate.rs")["lines"],
        "max_pyfunctions": 0,
    }


def test_growth_over_ceiling_is_flagged(tmp_path: Path) -> None:
    tool = _load_tool()
    _write_fake_target(tmp_path, lines=40, pyfunctions=10)
    ceiling: dict[str, Any] = {
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

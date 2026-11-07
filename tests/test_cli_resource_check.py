# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — resource-check command exit statuses

"""The ``resource-check`` command maps each overall verdict to its exit status."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.engine_requirement import require_engine

require_engine()

from sc_neurocore.cli.parser import main  # noqa: E402
from sc_neurocore.studio import resource_feedback  # noqa: E402
from tests.test_ir_sv_target_report import _dense  # noqa: E402
from tests.test_studio_resource_feedback import real_flow  # noqa: E402


@pytest.fixture
def graph_file(tmp_path: Path) -> Path:
    """Write a valid 2x2 dense graph as IR text."""
    path = tmp_path / "dense.sc.graph"
    path.write_text(_dense(2, 2).to_text(), encoding="utf-8")
    return path


def _fixed_verdict(verdict: str) -> Any:
    def run(graph: object, **kwargs: object) -> dict[str, Any]:
        return {
            "verdict": verdict,
            "checks": [
                {"name": "timing", "verdict": verdict, "detail": f"clock {kwargs['clock_mhz']} MHz"}
            ],
        }

    return run


@pytest.mark.parametrize(
    ("verdict", "status"), [("pass", 0), ("violation", 1), ("undetermined", 2)]
)
def test_each_verdict_has_its_exit_status(
    verdict: str,
    status: int,
    graph_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Pass exits 0, a violation 1, and an undetermined run 2, never 0."""
    monkeypatch.setattr(resource_feedback, "run_resource_feedback", _fixed_verdict(verdict))
    assert (
        main(["resource-check", str(graph_file), "--clock-mhz", "12", "--timeout-seconds", "900"])
        == status
    )
    out = capsys.readouterr().out
    assert f"verdict: {verdict}" in out and "timing: clock 12 MHz" in out


def test_report_is_written_when_an_output_is_given(
    graph_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The JSON report on disk is the report the verdict came from."""
    monkeypatch.setattr(resource_feedback, "run_resource_feedback", _fixed_verdict("violation"))
    output = tmp_path / "report.json"
    assert (
        main(
            [
                "resource-check",
                str(graph_file),
                "--clock-mhz",
                "12",
                "--timeout-seconds",
                "900",
                "-o",
                str(output),
            ]
        )
        == 1
    )
    assert json.loads(output.read_text(encoding="utf-8"))["verdict"] == "violation"


def test_unreadable_or_unparsable_graphs_and_a_zero_clock_are_undetermined(
    graph_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Input the command cannot judge exits 2 with a message, before any tool runs."""
    assert (
        main(
            [
                "resource-check",
                str(tmp_path / "missing.sc.graph"),
                "--clock-mhz",
                "12",
                "--timeout-seconds",
                "900",
            ]
        )
        == 2
    )
    assert "cannot read IR graph" in capsys.readouterr().out
    garbage = tmp_path / "garbage.sc.graph"
    garbage.write_text("this is not IR\n", encoding="utf-8")
    assert (
        main(["resource-check", str(garbage), "--clock-mhz", "12", "--timeout-seconds", "900"]) == 2
    )
    assert "cannot read IR graph" in capsys.readouterr().out
    assert (
        main(["resource-check", str(graph_file), "--clock-mhz", "0", "--timeout-seconds", "900"])
        == 2
    )
    assert "--clock-mhz and --timeout-seconds must be positive" in capsys.readouterr().out
    zero_timeout = [
        "resource-check",
        str(graph_file),
        "--clock-mhz",
        "12",
        "--timeout-seconds",
        "0",
    ]
    assert main(zero_timeout) == 2
    assert "--clock-mhz and --timeout-seconds must be positive" in capsys.readouterr().out


def test_a_graph_that_fails_verification_exits_undetermined(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A parsed graph with verification problems is listed and exits 2."""
    from tests.test_evidence_packet import _DCLS

    failing = tmp_path / "failing.sc.graph"
    failing.write_text(
        _DCLS.replace("bitstream<7>", "bool").replace("offsets=[6]", "offsets=[0;1]"),
        encoding="utf-8",
    )
    assert (
        main(["resource-check", str(failing), "--clock-mhz", "12", "--timeout-seconds", "900"]) == 2
    )
    out = capsys.readouterr().out
    assert "fails IR verification:" in out and out.count("\n  ") >= 1


def test_missing_tools_exit_undetermined(
    graph_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without Yosys and nextpnr the real command exits 2."""
    monkeypatch.setenv("PATH", str(tmp_path))
    assert (
        main(["resource-check", str(graph_file), "--clock-mhz", "12", "--timeout-seconds", "900"])
        == 2
    )


def test_only_the_calibrated_target_is_offered() -> None:
    """An unsupported target is an argument error, not a silent fallback."""
    with pytest.raises(SystemExit) as raised:
        main(
            [
                "resource-check",
                "g.sc.graph",
                "--clock-mhz",
                "12",
                "--timeout-seconds",
                "900",
                "--target",
                "ecp5",
            ]
        )
    assert raised.value.code == 2


@real_flow
def test_a_small_dense_layer_exits_zero_on_the_real_flow(graph_file: Path, tmp_path: Path) -> None:
    """2x2 dense at 12 MHz passes every check on Yosys and nextpnr."""
    output = tmp_path / "report.json"
    status = main(
        [
            "resource-check",
            str(graph_file),
            "--clock-mhz",
            "12",
            "--timeout-seconds",
            "900",
            "-o",
            str(output),
        ]
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0, report["checks"]
    assert report["verdict"] == "pass"

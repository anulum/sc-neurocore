# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (symbiyosys_integration) from former test_cli_formal.py

from __future__ import annotations

from tests.cli_formal_support import *  # noqa: F403


def test_formal_verify_network_records_missing_symbiyosys(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "formal"

    with mock.patch("sc_neurocore.cli.commands.formal.shutil.which", return_value=None):
        rc = run_cli(
            "formal",
            "verify-network",
            "--module-name",
            "dense_lif_frontier_fixture",
            "--input-width",
            "3",
            "--output-width",
            "1",
            "--state-width",
            "16",
            "--output-index",
            "0",
            "--window-cycles",
            "4",
            "--max-spikes",
            "2",
            "--run-symbiyosys",
            "--output",
            str(out_dir),
        )

    assert rc == 0
    report = json.loads((out_dir / "formal_rate_bound_report.json").read_text(encoding="utf-8"))
    assert report["symbiyosys"]["requested"] is True
    assert report["symbiyosys"]["status"] == "tool_unavailable"
    assert report["symbiyosys"]["returncode"] is None
    assert "SymbiYosys unavailable" in capsys.readouterr().out
    assert (out_dir / "dense_lif_frontier_fixture.sby").exists()


def test_formal_verify_network_runs_symbiyosys_when_available(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "formal"
    completed = subprocess.CompletedProcess(
        args=["/usr/bin/sby", "-f", str(out_dir / "dense_lif_frontier_fixture.sby")],
        returncode=0,
        stdout="PASS\n",
        stderr="",
    )

    with (
        mock.patch("sc_neurocore.cli.commands.formal.shutil.which", return_value="/usr/bin/sby"),
        mock.patch(
            "sc_neurocore.cli.commands.formal.subprocess.run", return_value=completed
        ) as m_run,
    ):
        rc = run_cli(
            "formal",
            "verify-network",
            "--module-name",
            "dense_lif_frontier_fixture",
            "--input-width",
            "3",
            "--output-width",
            "1",
            "--state-width",
            "16",
            "--output-index",
            "0",
            "--window-cycles",
            "4",
            "--max-spikes",
            "2",
            "--run-symbiyosys",
            "--output",
            str(out_dir),
        )

    assert rc == 0
    m_run.assert_called_once()
    assert m_run.call_args.args[0] == [
        "/usr/bin/sby",
        "-f",
        str(out_dir / "dense_lif_frontier_fixture.sby"),
    ]
    report = json.loads((out_dir / "formal_rate_bound_report.json").read_text(encoding="utf-8"))
    assert report["symbiyosys"]["status"] == "passed"
    assert report["symbiyosys"]["returncode"] == 0
    assert report["symbiyosys"]["stdout"] == "PASS\n"
    assert "SymbiYosys passed" in capsys.readouterr().out


def test_formal_verify_network_returns_nonzero_on_symbiyosys_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "formal"
    completed = subprocess.CompletedProcess(
        args=["/usr/bin/sby", "-f", str(out_dir / "dense_lif_frontier_fixture.sby")],
        returncode=1,
        stdout="FAIL\n",
        stderr="assert failed\n",
    )

    with (
        mock.patch("sc_neurocore.cli.commands.formal.shutil.which", return_value="/usr/bin/sby"),
        mock.patch("sc_neurocore.cli.commands.formal.subprocess.run", return_value=completed),
    ):
        rc = run_cli(
            "formal",
            "verify-network",
            "--module-name",
            "dense_lif_frontier_fixture",
            "--input-width",
            "3",
            "--output-width",
            "1",
            "--state-width",
            "16",
            "--output-index",
            "0",
            "--window-cycles",
            "4",
            "--max-spikes",
            "2",
            "--run-symbiyosys",
            "--output",
            str(out_dir),
        )

    assert rc == 1
    report = json.loads((out_dir / "formal_rate_bound_report.json").read_text(encoding="utf-8"))
    assert report["symbiyosys"]["status"] == "failed"
    assert report["symbiyosys"]["returncode"] == 1
    assert report["symbiyosys"]["stderr"] == "assert failed\n"
    assert "SymbiYosys failed" in capsys.readouterr().out

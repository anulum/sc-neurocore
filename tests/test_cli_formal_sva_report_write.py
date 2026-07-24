# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (sva_report_write) from former test_cli_formal.py

from __future__ import annotations

from tests.cli_formal_support import *  # noqa: F403


def test_formal_verify_network_writes_sva_and_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "formal"

    rc = run_cli(
        "formal",
        "verify-network",
        "--module-name",
        "dense_lif_frontier_fixture",
        "--input-width",
        "3",
        "--output-width",
        "2",
        "--state-width",
        "16",
        "--output-index",
        "0",
        "--window-cycles",
        "16",
        "--max-spikes",
        "3",
        "--output",
        str(out_dir),
    )

    assert rc == 0
    sva_path = out_dir / "dense_lif_frontier_fixture_rate_bound.sv"
    report_path = out_dir / "formal_rate_bound_report.json"
    assert "Formal network verification artifacts written" in capsys.readouterr().out
    assert "a_output0_rate_bound" in sva_path.read_text(encoding="utf-8")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == "sc-neurocore.formal-network-rate-bound.v0.1"
    assert report["network"]["name"] == "dense_lif_frontier_fixture"
    assert report["rate_bound"]["window_cycles"] == 16
    assert report["replay"] is None
    assert report["artifacts"]["sva"] == str(sva_path)
    rtl_path = out_dir / "dense_lif_frontier_fixture.v"
    assert report["artifacts"]["rtl"] == str(rtl_path)
    validate_formal_network_report(report, artifact_root=out_dir)
    assert "module dense_lif_frontier_fixture (" in rtl_path.read_text(encoding="utf-8")
    assert "dense_lif_frontier_fixture.v" in (out_dir / "dense_lif_frontier_fixture.sby").read_text(
        encoding="utf-8"
    )

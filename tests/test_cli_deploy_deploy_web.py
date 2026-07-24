# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (deploy_web) from former test_cli_deploy.py

from __future__ import annotations

from tests.cli_deploy_support import *  # noqa: F403


def test_deploy_web_generates_browser_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The web target dispatches to the browser deployment builder."""
    import sc_neurocore.edge.web_deploy as web_deploy

    manifest = types.SimpleNamespace(
        artefacts={"manifest": "deployment.json", "html": "index.html"}
    )
    builder = mock.Mock(return_value=manifest)
    monkeypatch.setattr(web_deploy, "build_web_deployment", builder)

    assert (
        run_cli(
            "deploy",
            str(tmp_path / "model.nir"),
            "--target",
            "web",
            "--output",
            str(tmp_path / "web"),
        )
        == 0
    )
    assert builder.call_count == 1
    output = capsys.readouterr().out
    assert "deployment.json" in output
    assert "index.html" in output


def test_deploy_web_reports_builder_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Browser deployment validation errors return status one."""
    import sc_neurocore.edge.web_deploy as web_deploy

    monkeypatch.setattr(
        web_deploy,
        "build_web_deployment",
        mock.Mock(side_effect=ValueError("invalid browser model")),
    )

    assert run_cli("deploy", str(tmp_path / "model.nir"), "--target", "web") == 1
    assert "invalid browser model" in capsys.readouterr().out

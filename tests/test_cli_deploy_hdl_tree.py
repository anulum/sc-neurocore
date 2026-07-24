# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (hdl_tree) from former test_cli_deploy.py

from __future__ import annotations

from tests.cli_deploy_support import *  # noqa: F403

def test_deploy_replaces_existing_hdl_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regeneration replaces a stale HDL directory atomically at command scope."""
    import sc_neurocore.nir_bridge as bridge

    model = tmp_path / "model.nir"
    model.write_bytes(b"fixture")
    output = tmp_path / "project"
    stale_hdl = output / "hdl"
    stale_hdl.mkdir(parents=True)
    (stale_hdl / "stale.v").write_text("stale", encoding="utf-8")
    fake_nir = fake_module("nir", read=mock.Mock(return_value=object()))
    monkeypatch.setattr(
        bridge,
        "from_nir",
        lambda _graph, *, dt: types.SimpleNamespace(topo_order=["input"]),
    )

    with mock.patch.dict("sys.modules", {"nir": fake_nir}):
        assert run_cli("deploy", str(model), "--target", "artix7", "--output", str(output)) == 0

    assert not (stale_hdl / "stale.v").exists()


def test_deploy_warns_when_hdl_tree_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An installed runtime without repository HDL reports the omitted copy explicitly."""
    import sc_neurocore.cli.commands.deploy as deploy_command
    import sc_neurocore.nir_bridge as bridge

    model = tmp_path / "model.nir"
    model.write_bytes(b"fixture")
    fake_nir = fake_module("nir", read=mock.Mock(return_value=object()))
    monkeypatch.setattr(deploy_command, "_find_hdl_source", lambda: None)
    monkeypatch.setattr(
        bridge,
        "from_nir",
        lambda _graph, *, dt: types.SimpleNamespace(topo_order=["input"]),
    )

    with mock.patch.dict("sys.modules", {"nir": fake_nir}):
        assert run_cli("deploy", str(model), "--target", "artix7") == 0

    assert "HDL source directory not found" in capsys.readouterr().out



# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (deploy_nir) from former test_cli_deploy.py

from __future__ import annotations

from tests.cli_deploy_support import *  # noqa: F403


def test_deploy_nir_writes_hardware_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A NIR graph traverses the public non-checkpoint deployment route."""
    import sc_neurocore.nir_bridge as bridge

    model = tmp_path / "model.nir"
    model.write_bytes(b"fixture")
    fake_nir = fake_module("nir", read=mock.Mock(return_value=object()))
    monkeypatch.setattr(
        bridge,
        "from_nir",
        lambda _graph, *, dt: types.SimpleNamespace(topo_order=["input", "lif"]),
    )

    with mock.patch.dict("sys.modules", {"nir": fake_nir}):
        assert (
            run_cli(
                "deploy",
                str(model),
                "--target",
                "artix7",
                "--output",
                str(tmp_path / "project"),
            )
            == 0
        )

    assert (tmp_path / "project" / "project.tcl").is_file()

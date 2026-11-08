# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (auto_synthesis) from former test_cli_deploy.py

"""Open-source synthesis, place and route, and packing reached from the deploy command."""

from __future__ import annotations

import shutil

from tests.cli_deploy_support import *  # noqa: F403


def test_deploy_reports_successful_open_source_synthesis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A successful Yosys adapter result is surfaced by the public deploy command."""
    import sc_neurocore.cli.commands.deploy as deploy_command
    import sc_neurocore.nir_bridge as bridge

    model = tmp_path / "model.nir"
    model.write_bytes(b"fixture")
    fake_nir = fake_module("nir", read=mock.Mock(return_value=object()))
    monkeypatch.setattr(deploy_command, "run_auto_synthesis", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        bridge,
        "from_nir",
        lambda _graph, *, dt: types.SimpleNamespace(topo_order=["input"]),
    )

    with mock.patch.dict("sys.modules", {"nir": fake_nir}):
        assert run_cli("deploy", str(model), "--target", "ice40") == 0

    assert "Synthesis succeeded" in capsys.readouterr().out


def test_deploy_reports_missing_open_source_synthesis_tool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A missing Yosys binary leaves an actionable manual command."""
    import sc_neurocore.cli.commands.deploy as deploy_command
    import sc_neurocore.nir_bridge as bridge

    model = tmp_path / "model.nir"
    model.write_bytes(b"fixture")
    fake_nir = fake_module("nir", read=mock.Mock(return_value=object()))
    monkeypatch.setattr(deploy_command, "run_auto_synthesis", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        bridge,
        "from_nir",
        lambda _graph, *, dt: types.SimpleNamespace(topo_order=["input"]),
    )

    with mock.patch.dict("sys.modules", {"nir": fake_nir}):
        assert run_cli("deploy", str(model), "--target", "ice40") == 0

    output = capsys.readouterr().out
    assert "Yosys not found" in output
    assert "make synth" in output


def test_run_auto_synthesis_returns_false_without_yosys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The synthesis adapter stays optional when Yosys is not installed."""
    import sc_neurocore.cli.commands.deploy as deploy_command
    import shutil

    monkeypatch.setattr(shutil, "which", lambda _program: None)
    assert not deploy_command.run_auto_synthesis(
        str(_synthesis_project(tmp_path)),
        "ice40",
        "fixture",
        deploy_command.TARGET_CONFIGS["ice40"],
    )


def test_run_auto_synthesis_reports_yosys_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Yosys stderr is reduced to a bounded diagnostic on failure."""
    import sc_neurocore.cli.commands.deploy as deploy_command
    import shutil

    monkeypatch.setattr(shutil, "which", lambda _program: "/tools/yosys")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["yosys"],
            returncode=1,
            stdout="",
            stderr="one\ntwo\nthree\nfour\nfive\nsix\n",
        ),
    )

    assert not deploy_command.run_auto_synthesis(
        str(_synthesis_project(tmp_path)),
        "ice40",
        "fixture",
        deploy_command.TARGET_CONFIGS["ice40"],
    )
    output = capsys.readouterr().out
    assert "Yosys synthesis failed" in output
    assert "one" not in output
    assert "six" in output


def test_run_auto_synthesis_succeeds_without_place_and_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A valid synthesis JSON is useful evidence even without nextpnr."""
    import sc_neurocore.cli.commands.deploy as deploy_command
    import shutil

    monkeypatch.setattr(
        shutil,
        "which",
        lambda program: "/tools/yosys" if program == "yosys" else None,
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["yosys"],
            returncode=0,
            stdout="banner\nNumber of cells: 12\n",
            stderr="",
        ),
    )

    assert deploy_command.run_auto_synthesis(
        str(_synthesis_project(tmp_path)),
        "ice40",
        "fixture",
        deploy_command.TARGET_CONFIGS["ice40"],
    )
    assert "Number of cells: 12" in capsys.readouterr().out


def test_run_auto_synthesis_reports_place_and_route_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A nextpnr failure preserves the successful synthesis artefact."""
    import sc_neurocore.cli.commands.deploy as deploy_command
    import shutil

    monkeypatch.setattr(
        shutil,
        "which",
        lambda program: f"/tools/{program}" if program in {"yosys", "nextpnr-ice40"} else None,
    )
    results = iter(
        [
            subprocess.CompletedProcess(args=["yosys"], returncode=0, stdout="", stderr=""),
            subprocess.CompletedProcess(args=["nextpnr"], returncode=1, stdout="", stderr="failed"),
        ]
    )
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: next(results))

    assert deploy_command.run_auto_synthesis(
        str(_synthesis_project(tmp_path)),
        "ice40",
        "fixture",
        deploy_command.TARGET_CONFIGS["ice40"],
    )
    assert "PnR failed" in capsys.readouterr().out


def test_run_auto_synthesis_succeeds_without_pack_tool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Place-and-route success does not require a bitstream packer."""
    import sc_neurocore.cli.commands.deploy as deploy_command
    import shutil

    monkeypatch.setattr(
        shutil,
        "which",
        lambda program: f"/tools/{program}" if program in {"yosys", "nextpnr-ice40"} else None,
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="",
            stderr="",
        ),
    )

    assert deploy_command.run_auto_synthesis(
        str(_synthesis_project(tmp_path)),
        "ice40",
        "fixture",
        deploy_command.TARGET_CONFIGS["ice40"],
    )


@pytest.mark.parametrize("create_bitstream", [False, True])
def test_run_auto_synthesis_invokes_pack_tool(
    create_bitstream: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The packer is invoked and any produced bitstream is measured."""
    import sc_neurocore.cli.commands.deploy as deploy_command
    import shutil

    output = _synthesis_project(tmp_path)
    monkeypatch.setattr(shutil, "which", lambda program: f"/tools/{program}")

    def run_tool(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[0].endswith("icepack") and create_bitstream:
            (Path(str(kwargs["cwd"])) / "fixture.bin").write_bytes(b"\x00" * 1024)
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run_tool)

    assert deploy_command.run_auto_synthesis(
        str(output),
        "ice40",
        "fixture",
        deploy_command.TARGET_CONFIGS["ice40"],
    )
    emitted = "Bitstream:" in capsys.readouterr().out
    assert emitted is create_bitstream


def test_find_hdl_source_returns_none_outside_source_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The HDL locator returns ``None`` when no ancestor owns an HDL directory."""
    import sc_neurocore.cli.commands.deploy as deploy_command

    monkeypatch.setattr(deploy_command, "Path", lambda _value: tmp_path / "orphan" / "cli.py")
    assert deploy_command._find_hdl_source() is None


@pytest.mark.skipif(
    not all(shutil.which(tool) for tool in ("yosys", "nextpnr-ice40")),
    reason="Yosys and nextpnr-ice40 are required",
)
def test_a_hierarchical_design_is_placed_and_routed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A flattened hierarchy reaches nextpnr without Yosys 0.70 $scopeinfo cells.

    Left in, nextpnr 0.6 stops with "no BELs remaining to implement cell type
    '$scopeinfo'" and the run prints "PnR failed".
    """
    import sc_neurocore.cli.commands.deploy as deploy_command

    (tmp_path / "hdl").mkdir()
    (tmp_path / "hdl" / "cell.v").write_text(
        "module cell(input wire clk, input wire d, output reg q);\n"
        "    always @(posedge clk) q <= ~d;\n"
        "endmodule\n",
        encoding="utf-8",
    )
    (tmp_path / "fixture.sv").write_text(
        "module fixture(input wire clk, input wire d, output wire q);\n"
        "    wire mid;\n"
        "    cell first(.clk(clk), .d(d), .q(mid));\n"
        "    cell second(.clk(clk), .d(mid), .q(q));\n"
        "endmodule\n",
        encoding="utf-8",
    )
    assert deploy_command.run_auto_synthesis(
        str(tmp_path), "ice40", "fixture", deploy_command.TARGET_CONFIGS["ice40"]
    )
    out = capsys.readouterr().out
    assert "PnR succeeded: fixture.asc" in out, out
    assert "PnR failed" not in out
    netlist = json.loads((tmp_path / "fixture.json").read_text(encoding="utf-8"))
    types = {c["type"] for m in netlist["modules"].values() for c in m.get("cells", {}).values()}
    assert "$scopeinfo" not in types


@pytest.mark.parametrize(
    ("target", "tools", "outputs"),
    [
        ("ice40", ("yosys", "nextpnr-ice40", "icepack"), ("fixture.asc", "fixture.bin")),
        ("ecp5", ("yosys", "nextpnr-ecp5", "ecppack"), ("fixture.config", "fixture.bit")),
    ],
)
def test_the_generated_makefile_builds_a_bitstream(
    target: str, tools: tuple[str, ...], outputs: tuple[str, ...], tmp_path: Path
) -> None:
    """``make bitstream`` in a deployment runs each family's own place-and-route and packer."""
    import sc_neurocore.cli.commands.deploy as deploy_command

    missing = [tool for tool in (*tools, "make") if shutil.which(tool) is None]
    if missing:
        pytest.skip(f"not installed: {', '.join(missing)}")
    (tmp_path / "hdl").mkdir()
    (tmp_path / "hdl" / "cell.v").write_text(
        "module cell(input wire clk, input wire d, output reg q);\n"
        "    always @(posedge clk) q <= ~d;\n"
        "endmodule\n",
        encoding="utf-8",
    )
    (tmp_path / "fixture.sv").write_text(
        "module fixture(input wire clk, input wire d, output wire q);\n"
        "    wire mid;\n"
        "    cell first(.clk(clk), .d(d), .q(mid));\n"
        "    cell second(.clk(clk), .d(mid), .q(q));\n"
        "endmodule\n",
        encoding="utf-8",
    )
    deploy_command._generate_project(str(tmp_path), target, "fixture")
    built = subprocess.run(
        ["make", "bitstream"], cwd=tmp_path, capture_output=True, text=True, timeout=600
    )
    assert built.returncode == 0, built.stdout[-2000:] + built.stderr[-2000:]
    for name in outputs:
        assert (tmp_path / name).stat().st_size > 0, name

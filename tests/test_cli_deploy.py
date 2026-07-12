# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Deployment CLI tests

"""Exercise checkpoint and target deployment through the public CLI."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import types
from unittest import mock

import pytest

from tests.cli_test_support import fake_module, run_cli


def _checkpoint_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_checkpoint_deploy(
    checkpoint: Path,
    output: Path,
    *,
    digest: str | None,
    target: str = "ice40",
    bitstream_length: int = 64,
) -> int:
    arguments = [
        "deploy",
        str(checkpoint),
        "--target",
        target,
        "--T",
        str(bitstream_length),
        "--output",
        str(output),
    ]
    if digest is not None:
        arguments.extend(("--checkpoint-sha256", digest))
    return run_cli(*arguments)


def test_deploy_without_model_reports_usage(capsys: pytest.CaptureFixture[str]) -> None:
    assert run_cli("deploy") == 1
    assert "deploy requires a model file" in capsys.readouterr().out


def test_deploy_rejects_unsupported_extension(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"\x00")
    assert run_cli("deploy", str(model), "--output", str(tmp_path / "out")) == 1
    assert "unsupported file format" in capsys.readouterr().out


def test_deploy_pytorch_writes_yosys_project(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 2),
    )
    checkpoint = tmp_path / "tiny.pt"
    torch.save(model.state_dict(), checkpoint)
    output = tmp_path / "deploy_out"

    assert (
        _run_checkpoint_deploy(
            checkpoint,
            output,
            digest=_checkpoint_digest(checkpoint),
        )
        == 0
    )
    assert (output / "sc_deploy_lif.sv").stat().st_size > 0
    assert (output / "Makefile").is_file()
    assert "ice40" in (output / "README.md").read_text(encoding="utf-8")
    power_model = json.loads((output / "power_thermal_model.json").read_text(encoding="utf-8"))
    assert power_model["source_mode"] == "pre_silicon_estimate"
    assert power_model["workload"]["layer_sizes"] == [[4, 8], [8, 2]]


def test_deploy_pytorch_writes_vivado_project(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    checkpoint = tmp_path / "tiny.pt"
    torch.save(model.state_dict(), checkpoint)
    output = tmp_path / "vivado_out"

    assert (
        _run_checkpoint_deploy(
            checkpoint,
            output,
            digest=_checkpoint_digest(checkpoint),
            target="artix7",
        )
        == 0
    )
    assert (output / "project.tcl").is_file()
    assert not (output / "Makefile").exists()
    assert "artix7" in (output / "README.md").read_text(encoding="utf-8")


def test_deploy_pytorch_requires_digest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "nohash.pt"
    torch.save(torch.nn.Linear(2, 2).state_dict(), checkpoint)

    assert _run_checkpoint_deploy(checkpoint, tmp_path / "out", digest=None) == 1
    assert "--checkpoint-sha256" in capsys.readouterr().out


@pytest.mark.parametrize("digest", ["abc123", "g" * 64])
def test_deploy_pytorch_rejects_invalid_digest(
    digest: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "badsha.pt"
    torch.save(torch.nn.Linear(2, 2).state_dict(), checkpoint)

    assert _run_checkpoint_deploy(checkpoint, tmp_path / "out", digest=digest) == 1
    assert "64 hexadecimal characters" in capsys.readouterr().out


def test_deploy_rejects_non_tensor_state_entry(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "bad_state.pt"
    torch.save({"layer.weight": [1, 2, 3]}, checkpoint)

    assert (
        _run_checkpoint_deploy(
            checkpoint,
            tmp_path / "out",
            digest=_checkpoint_digest(checkpoint),
        )
        == 1
    )
    assert "entries must be tensors" in capsys.readouterr().out


def test_deploy_rejects_checkpoint_without_dense_weights(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "conv_only.pt"
    torch.save({"conv.weight": torch.randn(8, 1, 3, 3)}, checkpoint)

    assert (
        _run_checkpoint_deploy(
            checkpoint,
            tmp_path / "out",
            digest=_checkpoint_digest(checkpoint),
        )
        == 1
    )
    assert "does not contain any 2D dense '.weight' tensors" in capsys.readouterr().out


def test_deploy_rejects_non_floating_dense_weight(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "int_dense.pt"
    torch.save({"layer.weight": torch.ones(4, 4, dtype=torch.int64)}, checkpoint)

    assert (
        _run_checkpoint_deploy(
            checkpoint,
            tmp_path / "out",
            digest=_checkpoint_digest(checkpoint),
        )
        == 1
    )
    assert "must use floating-point dtype" in capsys.readouterr().out


def test_deploy_rejects_empty_dense_weight(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "empty_dense.pt"
    torch.save({"layer.weight": torch.empty(0, 4)}, checkpoint)

    assert (
        _run_checkpoint_deploy(
            checkpoint,
            tmp_path / "out",
            digest=_checkpoint_digest(checkpoint),
        )
        == 1
    )
    assert "must have non-zero 2D shape" in capsys.readouterr().out


def test_deploy_rejects_non_finite_dense_weight(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    torch = pytest.importorskip("torch")
    weight = torch.randn(4, 4, dtype=torch.float32)
    weight[0, 0] = torch.nan
    checkpoint = tmp_path / "nan_dense.pt"
    torch.save({"layer.weight": weight}, checkpoint)

    assert (
        _run_checkpoint_deploy(
            checkpoint,
            tmp_path / "out",
            digest=_checkpoint_digest(checkpoint),
        )
        == 1
    )
    assert "contains non-finite values" in capsys.readouterr().out


def test_deploy_rejects_excessive_dense_parameter_count(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    torch = pytest.importorskip("torch")
    expanded_weight = torch.ones(1, dtype=torch.float32).expand(1, 20_000_001)
    checkpoint = tmp_path / "too_many_dense_params.pt"
    torch.save({"layer.weight": expanded_weight}, checkpoint)

    assert (
        _run_checkpoint_deploy(
            checkpoint,
            tmp_path / "out",
            digest=_checkpoint_digest(checkpoint),
        )
        == 1
    )
    assert "dense parameter count exceeds safety limit" in capsys.readouterr().out


def test_deploy_rejects_incompatible_dense_weight_chain(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "bad_chain.pt"
    torch.save(
        {
            "layer_a.weight": torch.randn(3, 4, dtype=torch.float32),
            "layer_b.weight": torch.randn(2, 5, dtype=torch.float32),
        },
        checkpoint,
    )

    assert (
        _run_checkpoint_deploy(
            checkpoint,
            tmp_path / "out",
            digest=_checkpoint_digest(checkpoint),
        )
        == 1
    )
    assert "not composition-compatible" in capsys.readouterr().out


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


def test_deploy_reports_checkpoint_trust_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Trusted-checkpoint loader failures remain controlled command errors."""
    import sc_neurocore.security.checkpoint_loading as checkpoint_loading

    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"fixture")
    monkeypatch.setattr(
        checkpoint_loading,
        "safe_load_checkpoint",
        mock.Mock(
            side_effect=checkpoint_loading.CheckpointTrustError("digest does not match fixture")
        ),
    )

    assert (
        _run_checkpoint_deploy(
            checkpoint,
            tmp_path / "out",
            digest="0" * 64,
        )
        == 1
    )
    assert "digest does not match fixture" in capsys.readouterr().out


@pytest.mark.parametrize("state", [[], {1: object()}])
def test_deploy_rejects_non_string_state_dictionary(
    state: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Checkpoint payloads must be dictionaries with string keys."""
    import sc_neurocore.security.checkpoint_loading as checkpoint_loading

    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"fixture")
    monkeypatch.setattr(checkpoint_loading, "safe_load_checkpoint", lambda *_args, **_kwargs: state)

    assert (
        _run_checkpoint_deploy(
            checkpoint,
            tmp_path / "out",
            digest="0" * 64,
        )
        == 1
    )
    assert "state_dict-like dictionary" in capsys.readouterr().out


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


def _synthesis_project(tmp_path: Path) -> Path:
    """Create the minimum HDL tree accepted by the synthesis adapter."""
    output = tmp_path / "synthesis"
    hdl = output / "hdl"
    hdl.mkdir(parents=True)
    (hdl / "fixture.v").write_text("module fixture; endmodule\n", encoding="utf-8")
    return output


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

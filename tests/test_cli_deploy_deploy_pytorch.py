# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (deploy_pytorch) from former test_cli_deploy.py

from __future__ import annotations

from tests.cli_deploy_support import *  # noqa: F403


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

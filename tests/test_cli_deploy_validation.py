# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (validation) from former test_cli_deploy.py

from __future__ import annotations

from tests.cli_deploy_support import *  # noqa: F403


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

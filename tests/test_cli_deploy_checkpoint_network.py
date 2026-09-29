# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Deploy exports the network its checkpoint holds

"""``sc-neurocore deploy`` converts and exports exactly the checkpoint's network.

Every case saves a real PyTorch checkpoint, runs the command and reloads the
exported network, comparing its digest with converting the original model
directly; the Studio case trains a real conversion run first.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sc_neurocore.conversion.converted_io import (
    CONVERTED_NETWORK_SCHEMA_VERSION,
    load_converted_network,
    save_converted_network,
)
from sc_neurocore.conversion.loss_report import converted_sha256
from tests.cli_test_support import run_cli

torch = pytest.importorskip("torch")


def _deploy(
    tmp_path: Path, payload: object, *extra: str, target: str = "ice40"
) -> tuple[int, Path]:
    """Save a checkpoint, deploy it and return the status and output directory."""
    checkpoint = tmp_path / "model.pt"
    torch.save(payload, checkpoint)
    output = tmp_path / "out"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    status = run_cli(
        "deploy",
        str(checkpoint),
        "--target",
        target,
        "--T",
        "16",
        "--checkpoint-sha256",
        digest,
        "--output",
        str(output),
        *extra,
    )
    return status, output


def _deep_chain() -> Any:
    """Eleven dense layers of distinct widths, so only the true order composes."""
    torch.manual_seed(3)
    widths = [5, 7, 6, 9, 4, 8, 3, 10, 6, 5, 7, 2]
    layers: list[Any] = []
    for width_in, width_out in zip(widths, widths[1:]):
        layers += [torch.nn.Linear(width_in, width_out), torch.nn.ReLU()]
    return torch.nn.Sequential(*layers[:-1])


class TestStateDict:
    def test_layers_keep_their_order_and_biases(self, tmp_path: Path) -> None:
        from sc_neurocore.conversion import convert

        model = _deep_chain()
        assert "10.weight" in model.state_dict() and "2.weight" in model.state_dict()
        status, output = _deploy(tmp_path, model.state_dict())
        assert status == 0
        exported = load_converted_network(output / "converted_network.npz")
        assert converted_sha256(exported) == converted_sha256(convert(model, T=16))
        assert all(bias is not None for bias in exported.biases)
        manifest = json.loads((output / "converted_network.json").read_text())
        assert manifest["source"] == "state_dict" and manifest["calibration"] == "unit"
        assert manifest["layer_sizes"][0] == [5, 7] and manifest["layer_sizes"][-1] == [7, 2]
        assert manifest["converted_sha256"] == converted_sha256(exported)
        assert "does not carry" in manifest["rtl"]
        readme = (output / "README.md").read_text()
        assert "converted_network.npz" in readme and "does\n  not carry" in readme

    def test_a_layer_without_a_bias_stays_without_one(self, tmp_path: Path) -> None:
        from sc_neurocore.conversion import convert

        model = torch.nn.Sequential(torch.nn.Linear(3, 2, bias=False))
        status, output = _deploy(tmp_path, model.state_dict())
        assert status == 0
        exported = load_converted_network(output / "converted_network.npz")
        assert exported.biases == [None]
        assert converted_sha256(exported) == converted_sha256(convert(model, T=16))

    def test_calibration_samples_set_the_thresholds_and_the_target_fit(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from sc_neurocore.conversion import convert

        torch.manual_seed(4)
        model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3))
        samples = np.random.default_rng(5).random((32, 4))
        np.save(tmp_path / "calibration.npy", samples)
        status, output = _deploy(
            tmp_path, model.state_dict(), "--calibration", str(tmp_path / "calibration.npy")
        )
        assert status == 0
        expected = convert(model, torch.as_tensor(samples, dtype=torch.float32), T=16)
        exported = load_converted_network(output / "converted_network.npz")
        assert converted_sha256(exported) == converted_sha256(expected)
        report = json.loads((output / "target_report.json").read_text())
        assert report["profile"]["name"] == "ice40" and report["samples"] == 32
        assert report["converted_sha256"] == converted_sha256(exported)
        assert "ice40 Q7.8" in capsys.readouterr().out

    def test_a_target_without_a_profile_skips_the_fit(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        np.save(tmp_path / "calibration.npy", np.full((4, 2), 0.5))
        status, output = _deploy(
            tmp_path,
            torch.nn.Linear(2, 2).state_dict(),
            "--calibration",
            str(tmp_path / "calibration.npy"),
            target="zynq",
        )
        assert status == 0 and not (output / "target_report.json").exists()
        assert "no fixed-point profile is registered for 'zynq'" in capsys.readouterr().out

    def test_no_calibration_names_what_it_would_measure(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        status, output = _deploy(tmp_path, torch.nn.Linear(2, 2).state_dict())
        assert status == 0 and not (output / "target_report.json").exists()
        assert "pass --calibration samples.npy" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "payload,samples,message",
        [
            (
                torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.BatchNorm1d(3)).state_dict(),
                None,
                "parameters a dense ReLU chain does not have",
            ),
            (
                {"0.weight": torch.ones(3, 2), "0.bias": torch.ones(2)},
                None,
                "must be a finite float vector of its outputs",
            ),
            (
                {"0.weight": torch.ones(3, 2), "0.bias": torch.tensor([1.0, float("nan"), 0.0])},
                None,
                "must be a finite float vector of its outputs",
            ),
            (torch.nn.Linear(2, 2).state_dict(), np.ones((3, 5)), "non-empty (samples, 2)"),
            (torch.nn.Linear(2, 2).state_dict(), np.full((3, 2), 4.0), "finite values in [0, 1]"),
        ],
    )
    def test_a_checkpoint_it_cannot_rebuild_exactly_is_refused(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        payload: object,
        samples: np.ndarray | None,
        message: str,
    ) -> None:
        extra: tuple[str, ...] = ()
        if samples is not None:
            np.save(tmp_path / "calibration.npy", samples)
            extra = ("--calibration", str(tmp_path / "calibration.npy"))
        status, _ = _deploy(tmp_path, payload, *extra)
        assert status == 1
        assert message in capsys.readouterr().out

    def test_an_unreadable_calibration_file_is_refused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        status, _ = _deploy(
            tmp_path, torch.nn.Linear(2, 2).state_dict(), "--calibration", str(tmp_path / "no.npy")
        )
        assert status == 1 and "Error:" in capsys.readouterr().out


@pytest.fixture(scope="module")
def studio_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A finished Studio conversion run's sandbox."""
    from sc_neurocore.studio.platform import StudioJobContext
    from sc_neurocore.studio.training import TrainingJob

    root = tmp_path_factory.mktemp("studio-run")
    context = StudioJobContext(
        job_id="sj_deploy_source",
        work_dir=root,
        cancel_event=threading.Event(),
        max_artifact_bytes=1 << 22,
    )
    config = {
        "model_kind": "qcfs_conversion",
        "dataset": "synthetic",
        "epochs": 1,
        "batch_size": 32,
        "timesteps": 4,
        "hidden": [16],
        "seed": 5,
    }
    assert TrainingJob(config, job_id=context.job_id).run_blocking(context)["training_status"] == (
        "completed"
    )
    return root


class TestStudioCheckpoint:
    def test_a_conversion_run_deploys_the_network_it_measured(
        self, tmp_path: Path, studio_run: Path
    ) -> None:
        payload = torch.load(studio_run / "training/model_state.pt", weights_only=True)
        status, output = _deploy(tmp_path, payload)
        assert status == 0
        manifest = json.loads((output / "converted_network.json").read_text())
        report = json.loads((studio_run / "training/conversion_report.json").read_text())
        assert manifest["source"] == "studio_qcfs_conversion"
        assert manifest["calibration"] == "learned QCFS thresholds"
        assert manifest["T"] == 4
        assert manifest["converted_sha256"] == report["converted_sha256"]

    def test_a_spiking_run_is_not_reinterpreted_as_an_ann(
        self, tmp_path: Path, studio_run: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        payload = torch.load(studio_run / "training/model_state.pt", weights_only=True)
        payload["config"] = {"dataset": "synthetic", "hidden": [16]}
        status, _ = _deploy(tmp_path, payload)
        assert status == 1
        assert "already a spiking network" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "change,message",
        [
            ({"config": {"model_kind": "qcfs_conversion", "hidden": [8]}}, "do not match"),
            ({"config": None}, "must carry its config and model_state_dict"),
            ({"model_state_dict": None}, "must carry its config and model_state_dict"),
        ],
    )
    def test_a_studio_checkpoint_that_disagrees_with_itself_is_refused(
        self,
        tmp_path: Path,
        studio_run: Path,
        capsys: pytest.CaptureFixture[str],
        change: dict[str, Any],
        message: str,
    ) -> None:
        payload = torch.load(studio_run / "training/model_state.pt", weights_only=True)
        status, _ = _deploy(tmp_path, {**payload, **change})
        assert status == 1
        assert message in capsys.readouterr().out

    def test_the_schema_matches_the_studio_seal(self) -> None:
        from sc_neurocore.conversion.checkpoint_network import STUDIO_CHECKPOINT_SCHEMA_VERSION
        from sc_neurocore.studio.platform.training_weights import (
            STUDIO_TRAINING_TORCH_STATE_DICT_SCHEMA_VERSION,
        )

        assert STUDIO_CHECKPOINT_SCHEMA_VERSION == STUDIO_TRAINING_TORCH_STATE_DICT_SCHEMA_VERSION


class TestNetworkFile:
    def test_a_written_network_reads_back_identically(self, tmp_path: Path) -> None:
        from sc_neurocore.conversion import ConvertedSNN

        snn = ConvertedSNN(
            [np.eye(2), [[0.5, -0.25]]],
            [None, [0.125]],
            [1.0, 1.0],
            8,
            0.5,
            2.0,
            "linear",
            layer_membrane_fractions=[0.5, 0.0],
        )
        digest = save_converted_network(snn, tmp_path / "net.npz")
        again = load_converted_network(tmp_path / "net.npz")
        assert converted_sha256(again) == digest == converted_sha256(snn)
        np.testing.assert_array_equal(
            again.run(np.full((3, 2), 0.5), input_mode="constant"),
            snn.run(np.full((3, 2), 0.5), input_mode="constant"),
        )

    def _rewrite(self, tmp_path: Path, **change: Any) -> Path:
        """Write a valid file, then rewrite it with arrays changed."""
        from sc_neurocore.conversion import ConvertedSNN

        save_converted_network(
            ConvertedSNN([np.eye(2)], [[0.1, 0.2]], [1.0], 4), tmp_path / "a.npz"
        )
        with np.load(tmp_path / "a.npz", allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        arrays.update(change)
        arrays = {name: value for name, value in arrays.items() if value is not None}
        np.savez(tmp_path / "b.npz", **arrays)
        return tmp_path / "b.npz"

    @pytest.mark.parametrize(
        "change,message",
        [
            ({"weight_0": np.eye(2) * 2}, "digest does not match"),
            ({"bias_0": None}, "arrays do not match its header"),
            ({"extra": np.zeros(1)}, "arrays do not match its header"),
            ({"header": None}, "no header"),
            ({"header": np.array(json.dumps({"schema_version": "v0"}))}, "another schema"),
        ],
    )
    def test_a_changed_file_is_refused(
        self, tmp_path: Path, change: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            load_converted_network(self._rewrite(tmp_path, **change))

    def test_the_file_names_its_schema(self, tmp_path: Path) -> None:
        from sc_neurocore.conversion import ConvertedSNN

        save_converted_network(ConvertedSNN([np.eye(1)], [None], [1.0], 2), tmp_path / "n.npz")
        with np.load(tmp_path / "n.npz", allow_pickle=False) as data:
            header = json.loads(str(data["header"]))
        assert header["schema_version"] == CONVERTED_NETWORK_SCHEMA_VERSION

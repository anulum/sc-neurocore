# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Measured conversion loss acceptance

"""A conversion loss report states what both networks did on the same labelled samples.

Every case trains or builds a real PyTorch source, converts it with the public
converter and compares the report against an independent recomputation from
the two networks' own outputs.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from sc_neurocore.conversion import (  # noqa: E402
    ConversionLossReport,
    convert,
    measure_conversion_loss,
    replace_relu_with_qcfs,
)
from sc_neurocore.conversion.if_dispatch import resolve_replay_backend  # noqa: E402
from sc_neurocore.conversion.loss_report import (  # noqa: E402
    LOSS_REPORT_SCHEMA_VERSION,
    converted_sha256,
    data_sha256,
    source_sha256,
)

_NATIVE = (
    "SC_NEUROCORE_IF_RUST_LIB",
    "SC_NEUROCORE_IF_GO_LIB",
    "SC_NEUROCORE_IF_MOJO_LIB",
    "SC_NEUROCORE_IF_JULIA_ENABLED",
    "SC_NEUROCORE_IF_BENCHMARK",
)


@pytest.fixture(autouse=True)
def numpy_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every case with no native replay configured."""
    for variable in _NATIVE:
        monkeypatch.delenv(variable, raising=False)


def _blobs(samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Three separable classes of 6-feature points inside the unit cube."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 3, samples)
    centres = np.array([[0.2] * 6, [0.5] * 6, [0.8] * 6])
    inputs = np.clip(centres[labels] + rng.normal(0, 0.06, (samples, 6)), 0, 1)
    return inputs, labels


@pytest.fixture(scope="module")
def trained() -> nn.Module:
    """A QCFS MLP trained on the blobs until the source separates them."""
    torch.manual_seed(7)
    model = replace_relu_with_qcfs(
        nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 3)), T=8, theta=1.0
    )
    inputs, labels = _blobs(600, seed=1)
    x = torch.as_tensor(inputs, dtype=torch.float32)
    y = torch.as_tensor(labels)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.03)
    for _ in range(300):
        optimiser.zero_grad()
        nn.functional.cross_entropy(model(x), y).backward()
        optimiser.step()
    return model


def _independent(model: nn.Module, snn: object, inputs: np.ndarray, labels: np.ndarray) -> dict:
    """Recompute the report's measurements directly from both networks."""
    with torch.no_grad():
        source = model.eval()(torch.as_tensor(inputs, dtype=torch.float32)).double().numpy()
    response = snn.run(inputs.reshape(len(inputs), -1), input_mode="constant", backend="numpy")
    rates = response / snn.T * snn.output_scale
    return {
        "source_accuracy": float(np.mean(source.argmax(1) == labels)),
        "converted_accuracy": float(np.mean(response.argmax(1) == labels)),
        "agreement": float(np.mean(source.argmax(1) == response.argmax(1))),
        "rate_max_abs_error": float(np.abs(rates - source).max()),
    }


class TestMeasurement:
    def test_a_trained_qcfs_network_is_reported_as_measured(self, trained: nn.Module) -> None:
        inputs, labels = _blobs(250, seed=2)
        snn = convert(trained)
        report = measure_conversion_loss(trained, snn, inputs, labels, batch_size=64)
        expected = _independent(trained, snn, inputs, labels)
        assert report.schema_version == LOSS_REPORT_SCHEMA_VERSION
        assert (report.samples, report.classes, report.timesteps) == (250, 3, 8)
        assert report.backend == "numpy" and report.input_mode == "constant"
        assert report.source_accuracy == expected["source_accuracy"] > 0.9
        assert report.converted_accuracy == expected["converted_accuracy"]
        assert report.accuracy_drop == report.source_accuracy - report.converted_accuracy
        assert report.agreement == expected["agreement"]
        assert report.rate_max_abs_error == pytest.approx(expected["rate_max_abs_error"])
        assert 0 <= report.rate_mean_abs_error <= report.rate_max_abs_error
        public = report.to_public_dict()
        assert public["converted_sha256"] == converted_sha256(snn)
        assert ConversionLossReport(**public) == report

    def test_an_exact_conversion_reports_no_loss(self) -> None:
        torch.manual_seed(3)
        model = nn.Sequential(nn.Linear(5, 4)).double()
        inputs = np.random.default_rng(4).random((40, 5))
        with torch.no_grad():
            labels = model(torch.as_tensor(inputs)).argmax(1).numpy()
        report = measure_conversion_loss(model, convert(model), inputs, labels, batch_size=7)
        assert report.source_accuracy == report.converted_accuracy == report.agreement == 1.0
        assert report.accuracy_drop == 0.0
        assert report.rate_max_abs_error < 1e-12

    def test_source_shaped_inputs_are_flattened_for_the_converted_network(self) -> None:
        torch.manual_seed(5)
        model = nn.Sequential(nn.Flatten(), nn.Linear(6, 3))
        inputs = np.random.default_rng(6).random((12, 2, 3))
        report = measure_conversion_loss(model, convert(model), inputs, np.zeros(12, np.int64))
        assert report.samples == 12 and report.rate_max_abs_error < 1e-5

    def test_every_training_flag_is_restored(self, trained: nn.Module) -> None:
        inputs, labels = _blobs(20, seed=3)
        trained.train()
        trained[1].eval()
        measure_conversion_loss(trained, convert(trained), inputs, labels)
        assert trained.training and trained[0].training and not trained[1].training
        trained.eval()

    def test_poisson_batches_use_consecutive_wrapping_seeds(self, trained: nn.Module) -> None:
        inputs, labels = _blobs(5, seed=9)
        snn = convert(trained)
        seed = 2**32 - 2
        report = measure_conversion_loss(
            trained, snn, inputs, labels, input_mode="poisson", seed=seed, batch_size=2
        )
        seeds = [seed, 2**32 - 1, 0]
        responses = np.concatenate(
            [
                snn.run(inputs[k * 2 : k * 2 + 2], input_mode="poisson", seed=s, backend="numpy")
                for k, s in enumerate(seeds)
            ]
        )
        assert report.converted_accuracy == float(np.mean(responses.argmax(1) == labels))
        assert report.seed == seed and report.input_mode == "poisson"


class TestDigests:
    def test_each_digest_binds_its_own_object(self, trained: nn.Module) -> None:
        inputs, labels = _blobs(30, seed=4)
        snn = convert(trained)
        before = measure_conversion_loss(trained, snn, inputs, labels)
        assert before.source_sha256 == source_sha256(trained)
        assert before.data_sha256 == data_sha256(inputs.astype(np.float64), labels.astype(np.int64))
        changed_labels = labels.copy()
        changed_labels[0] = (changed_labels[0] + 1) % 3
        assert measure_conversion_loss(trained, snn, inputs, changed_labels).data_sha256 != (
            before.data_sha256
        )
        snn.weights[0][0, 0] += 1e-9
        assert converted_sha256(snn) != before.converted_sha256
        copy = replace_relu_with_qcfs(
            nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 3)), T=8
        )
        copy.load_state_dict(trained.state_dict())
        assert source_sha256(copy) == before.source_sha256
        with torch.no_grad():
            copy[2].bias[0] += 1e-6
        assert source_sha256(copy) != before.source_sha256

    def test_the_data_digest_binds_the_source_shape(self) -> None:
        labels = np.zeros(2, np.int64)
        flat = np.zeros((2, 6))
        assert data_sha256(flat, labels) != data_sha256(flat.reshape(2, 2, 3), labels)

    def test_an_absent_bias_differs_from_a_zero_bias(self) -> None:
        model = nn.Linear(3, 2, bias=False)
        with_zero = nn.Linear(3, 2)
        with torch.no_grad():
            with_zero.weight.copy_(model.weight)
            with_zero.bias.zero_()
        absent, zero = convert(model), convert(with_zero)
        assert absent.biases == [None]
        assert converted_sha256(absent) != converted_sha256(zero)

    def test_dtypes_numpy_lacks_are_digested(self) -> None:
        model = nn.Linear(3, 2).to(torch.bfloat16)
        assert len(source_sha256(model)) == 64
        assert source_sha256(model) != source_sha256(model.float())


@pytest.mark.parametrize(
    "change,message",
    [
        ({"batch_size": 0}, "batch_size"),
        ({"batch_size": True}, "batch_size"),
        ({"seed": -1}, "seed"),
        ({"seed": 2**32}, "seed"),
        ({"input_mode": "burst"}, "input_mode"),
        ({"inputs": np.zeros((0, 6))}, "at least one sample"),
        ({"inputs": np.zeros(6)}, "at least one sample"),
        ({"inputs": np.full((4, 6), np.nan)}, "finite"),
        ({"labels": np.zeros(4)}, "one integer class"),
        ({"labels": np.zeros(3, np.int64)}, "one integer class"),
        ({"labels": np.array([0, 1, 2, 3])}, r"\[0, classes\)"),
        ({"labels": np.array([0, 1, 2, -1])}, r"\[0, classes\)"),
        ({"inputs": np.full((4, 6), 2.0)}, "between zero and one"),
        ({"backend": "cuda"}, "unsupported dense IF backend"),
    ],
)
def test_an_unmeasurable_request_is_refused(trained: nn.Module, change: dict, message: str) -> None:
    request = {"inputs": np.full((4, 6), 0.5), "labels": np.array([0, 1, 2, 0]), **change}
    inputs, labels = request.pop("inputs"), request.pop("labels")
    with pytest.raises(ValueError, match=message):
        measure_conversion_loss(trained, convert(trained), inputs, labels, **request)


def test_networks_of_different_widths_are_refused(trained: nn.Module) -> None:
    other = convert(nn.Sequential(nn.Linear(6, 2)))
    with pytest.raises(ValueError, match="output width"):
        measure_conversion_loss(trained, other, np.full((2, 6), 0.5), np.array([0, 1]))


def test_a_source_without_one_vector_per_sample_is_refused() -> None:
    model = nn.Sequential(nn.Flatten(0), nn.Linear(12, 3))
    with pytest.raises(ValueError, match="one output vector per sample"):
        measure_conversion_loss(
            model, convert(nn.Linear(6, 3)), np.full((2, 6), 0.5), np.array([0, 1])
        )


class TestExecutedRuntime:
    def test_auto_names_the_floor_without_native_configuration(self) -> None:
        assert resolve_replay_backend("auto") == "numpy"
        for name in ("numpy", "rust", "go", "mojo", "julia"):
            assert resolve_replay_backend(name) == name
        unsupported: Any = "cuda"
        with pytest.raises(ValueError, match="unsupported"):
            resolve_replay_backend(unsupported)

    def test_a_malformed_julia_opt_in_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SC_NEUROCORE_IF_JULIA_ENABLED", "yes")
        with pytest.raises(RuntimeError, match="0 or 1"):
            resolve_replay_backend("auto")

    def test_auto_follows_the_static_order_without_loading(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("SC_NEUROCORE_IF_JULIA_ENABLED", "1")
        assert resolve_replay_backend("auto") == "julia"
        monkeypatch.setenv("SC_NEUROCORE_IF_MOJO_LIB", str(tmp_path / "absent.so"))
        assert resolve_replay_backend("auto") == "mojo"
        monkeypatch.setenv("SC_NEUROCORE_IF_GO_LIB", str(tmp_path / "absent.so"))
        assert resolve_replay_backend("auto") == "go"
        monkeypatch.setenv("SC_NEUROCORE_IF_RUST_LIB", str(tmp_path / "absent.so"))
        assert resolve_replay_backend("auto") == "rust"

    def test_a_configured_native_runtime_is_named_and_agrees(
        self, trained: nn.Module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        library = tmp_path / "if-replay.so"
        subprocess.run(
            ["go", "build", "-buildmode=c-shared", "-o", str(library), "./conversion/cshared"],
            cwd=Path(__file__).resolve().parents[1] / "src/sc_neurocore/accel/go",
            env=dict(os.environ, CGO_ENABLED="1"),
            capture_output=True,
            check=True,
            timeout=300,
        )
        inputs, labels = _blobs(40, seed=5)
        snn = convert(trained)
        floor = measure_conversion_loss(trained, snn, inputs, labels, batch_size=16)
        monkeypatch.setenv("SC_NEUROCORE_IF_GO_LIB", str(library))
        native = measure_conversion_loss(trained, snn, inputs, labels, batch_size=16)
        assert native.backend == "go"
        assert native.to_public_dict() == {**floor.to_public_dict(), "backend": "go"}

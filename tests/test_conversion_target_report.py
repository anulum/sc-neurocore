# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Target fixed-point compatibility and calibration acceptance

"""A target report fits a converted network into a real profile's format and measures it.

Profiles come from the registry or are built as real ``HardwareProfile``
values for formats small enough to force each refusal; every figure is checked
against the network's own replay.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from sc_neurocore.compiler.platforms import get_profile
from sc_neurocore.compiler.platforms.registry import HardwareProfile
from sc_neurocore.conversion import ConvertedSNN
from sc_neurocore.conversion.loss_report import converted_sha256
from sc_neurocore.conversion.target_report import (
    TARGET_REPORT_SCHEMA_VERSION,
    calibrate_for_target,
)


def _format(width: int, fraction: int, *, signed: bool = True) -> HardwareProfile:
    """A real profile of the given fixed-point format."""
    return HardwareProfile(
        name=f"q{width}_{fraction}",
        vendor="test",
        family="format",
        platform_class="asic",
        data_width=width,
        fraction=fraction,
        signed=signed,
    )


@pytest.fixture(scope="module")
def trained() -> tuple[ConvertedSNN, np.ndarray, np.ndarray]:
    """A trained QCFS classifier converted to IF, with its evaluation samples."""
    torch = pytest.importorskip("torch")
    from sc_neurocore.conversion import convert, replace_relu_with_qcfs

    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 3, 300)
    centres = np.array([[0.2] * 6, [0.5] * 6, [0.8] * 6])
    inputs = np.clip(centres[labels] + rng.normal(0, 0.06, (300, 6)), 0, 1)
    model = replace_relu_with_qcfs(
        torch.nn.Sequential(torch.nn.Linear(6, 16), torch.nn.ReLU(), torch.nn.Linear(16, 3)), T=8
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=0.03)
    x = torch.as_tensor(inputs, dtype=torch.float32)
    y = torch.as_tensor(labels)
    for _ in range(300):
        optimiser.zero_grad()
        torch.nn.functional.cross_entropy(model(x), y).backward()
        optimiser.step()
    return convert(model), inputs, labels


class TestRegistryTargets:
    def test_a_wide_format_keeps_the_network(self, trained: Any) -> None:
        snn, inputs, labels = trained
        report = calibrate_for_target(snn, get_profile("loihi2"), inputs, labels, batch_size=64)
        assert report.schema_version == TARGET_REPORT_SCHEMA_VERSION
        assert report.compatible and report.refusals == []
        assert report.profile["q_format"] == "Q11.12" and report.profile["overflow"] == "wrap"
        assert report.samples == 300 and report.timesteps == 8 and report.backend == "numpy"
        assert report.quantized_accuracy == report.float_accuracy
        assert report.agreement == 1.0 and report.exact_accumulation
        assert report.converted_sha256 == converted_sha256(snn)
        assert [layer.drive for layer in report.layers] == ["analog", "spikes"]
        assert [layer.readout for layer in report.layers] == ["spiking", "linear"]
        json.dumps(report.to_public_dict(), allow_nan=False)

    def test_each_layer_sits_at_its_finest_fitting_scale(self, trained: Any) -> None:
        snn, inputs, labels = trained
        profile = get_profile("truenorth")
        report = calibrate_for_target(snn, profile, inputs, labels)
        top = (2 ** (profile.data_width - 1) - 1) * 2.0**-profile.fraction
        grid = 2.0**-profile.fraction
        for layer, weight in zip(report.layers, snn.weights):
            scale = 2.0**layer.scale_exponent
            extent = max(
                np.abs(weight).max(), layer.measured_peak, 0 if layer.readout == "linear" else 1
            )
            assert extent * scale <= top < extent * scale * 2
            assert layer.weight_max_abs_error <= grid / scale / 2
            assert layer.zeroed_weights == int(
                np.count_nonzero((weight != 0) & (np.rint(weight * scale / grid) == 0))
            )
        assert report.layers[0].threshold_code == 2 ** (report.layers[0].scale_exponent + 7)
        assert report.layers[1].threshold_code == 0

    def test_without_labels_only_agreement_is_reported(self, trained: Any) -> None:
        snn, inputs, _ = trained
        report = calibrate_for_target(snn, get_profile("ecp5"), inputs)
        assert report.float_accuracy is report.quantized_accuracy is report.accuracy_drop is None
        assert 0 <= report.agreement <= 1


class TestExactness:
    def test_coefficients_on_the_grid_replay_unchanged(self) -> None:
        snn = ConvertedSNN(
            [[[0.25, 0.5], [0.75, -0.25]], [[0.5, 0.125]]],
            [[0.0625, 0.0], [0.25]],
            [1.0, 1.0],
            16,
            output_mode="linear",
        )
        inputs = np.random.default_rng(2).random((20, 2))
        report = calibrate_for_target(snn, _format(16, 8), inputs)
        assert report.output_max_abs_difference == 0.0 and report.agreement == 1.0
        assert all(layer.weight_max_abs_error == 0 for layer in report.layers)
        assert all(layer.bias_max_abs_error == 0 for layer in report.layers)

    def test_an_off_grid_preload_is_not_called_exact(self) -> None:
        snn = ConvertedSNN([[[1.0]], [[1.2]]], [None, None], [1.0, 1.0], 6, 0.5)
        report = calibrate_for_target(snn, _format(3, 0), np.ones((2, 1)))
        assert [layer.preload_exact for layer in report.layers] == [True, False]
        assert not report.exact_accumulation and report.compatible

    def test_a_format_wider_than_float64_is_not_called_exact(self, trained: Any) -> None:
        snn, inputs, _ = trained
        assert not calibrate_for_target(snn, _format(64, 20), inputs[:10]).exact_accumulation


class TestRefusals:
    def test_a_rounded_up_readout_that_overflows_is_counted(self) -> None:
        snn = ConvertedSNN([[[1.6]]], [None], [1.0], 4, output_mode="linear")
        report = calibrate_for_target(snn, _format(4, 0), np.ones((3, 1)))
        assert not report.compatible
        assert report.layers[0].overflow_steps == 3
        assert report.refusals == ["layer 0: membrane left the range at 3 sample-steps"]

    def test_a_dynamic_range_wider_than_the_format_is_refused(self) -> None:
        snn = ConvertedSNN([[[1000.0]], [[1.0]]], [None, None], [1.0, 1.0], 2)
        report = calibrate_for_target(snn, _format(4, 0), np.ones((1, 1)))
        assert "layer 0: its dynamic range leaves the threshold below one step" in report.refusals
        assert report.layers[0].threshold_code == 0

    def test_negative_coefficients_in_an_unsigned_format_are_refused(self) -> None:
        snn = ConvertedSNN([[[-0.5]]], [[0.25]], [1.0], 4)
        report = calibrate_for_target(snn, _format(8, 4, signed=False), np.ones((1, 1)))
        assert "layer 0: negative coefficients in an unsigned format" in report.refusals
        assert report.profile["signed"] is False

    def test_a_negative_bias_alone_is_refused_in_an_unsigned_format(self) -> None:
        snn = ConvertedSNN([[[0.5]]], [[-0.25]], [1.0], 4)
        report = calibrate_for_target(snn, _format(8, 4, signed=False), np.ones((1, 1)))
        assert "layer 0: negative coefficients in an unsigned format" in report.refusals

    def test_a_silent_layer_has_no_headroom_figure(self) -> None:
        snn = ConvertedSNN([[[0.0]], [[0.0]]], [None, None], [1.0, 1.0], 3, output_mode="linear")
        report = calibrate_for_target(snn, _format(8, 4), np.zeros((2, 1)))
        assert report.layers[1].measured_peak == 0.0 and report.layers[1].headroom_bits is None
        assert report.layers[1].scale_exponent == 0


@pytest.mark.parametrize(
    "change,message",
    [
        ({"batch_size": 0}, "batch_size"),
        ({"batch_size": True}, "batch_size"),
        ({"inputs": np.zeros((0, 1))}, "non-empty"),
        ({"inputs": np.zeros(3)}, "non-empty"),
        ({"inputs": np.full((2, 1), 1.5)}, r"\[0, 1\]"),
        ({"inputs": np.full((2, 1), np.nan)}, r"\[0, 1\]"),
        ({"labels": np.zeros(2)}, "one integer class"),
        ({"labels": np.zeros(3, np.int64)}, "one integer class"),
    ],
)
def test_an_uncalibratable_request_is_refused(change: dict[str, Any], message: str) -> None:
    request: dict[str, Any] = {"inputs": np.ones((2, 1)), "labels": None, **change}
    inputs, labels = request.pop("inputs"), request.pop("labels")
    snn = ConvertedSNN([[[0.5]]], [None], [1.0], 4)
    with pytest.raises(ValueError, match=message):
        calibrate_for_target(snn, _format(8, 4), inputs, labels, **request)

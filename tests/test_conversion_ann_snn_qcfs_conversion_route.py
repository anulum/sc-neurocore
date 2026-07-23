# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestQcfsConversionRoute from former test_conversion_ann_snn.py

"""Focused suite: TestQcfsConversionRoute from former test_conversion_ann_snn.py."""

from __future__ import annotations

from tests.conversion_ann_snn_support import *  # noqa: F403

class TestQcfsConversionRoute:
    def _qcfs_model(self, theta: float = 1.0, qcfs_t: int = 4) -> nn.Module:
        torch.manual_seed(7)
        return nn.Sequential(
            nn.Linear(4, 8),
            QCFSActivation(T=qcfs_t, theta=theta),
            nn.Linear(8, 3),
        )

    def test_qcfs_route_sets_membrane_shift(self) -> None:
        snn = convert(self._qcfs_model())
        assert snn.initial_membrane_fraction == 0.5

    def test_relu_route_has_no_membrane_shift(self) -> None:
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
        snn = convert(model)
        assert snn.initial_membrane_fraction == 0.0

    def test_qcfs_route_infers_timesteps_from_layers(self) -> None:
        snn = convert(self._qcfs_model(qcfs_t=5))
        assert snn.T == 5

    def test_relu_route_defaults_timesteps_to_sixteen(self) -> None:
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
        snn = convert(model)
        assert snn.T == 16

    def test_explicit_timesteps_override_qcfs_inference(self) -> None:
        snn = convert(self._qcfs_model(qcfs_t=4), T=32)
        assert snn.T == 32

    def test_qcfs_route_normalises_thresholds_to_unity(self) -> None:
        snn = convert(self._qcfs_model(theta=2.5))
        assert snn.thresholds == [1.0, 1.0]

    def test_qcfs_theta_scales_first_layer_weights(self) -> None:
        model = self._qcfs_model(theta=4.0)
        raw = model[0].weight.detach().cpu().numpy().copy()
        snn = convert(model)
        # First layer is divided by its QCFS threshold (theta == 4.0).
        np.testing.assert_allclose(snn.weights[0], raw / 4.0, rtol=1e-6)

    def test_qcfs_route_ignores_calibration_data(self) -> None:
        model = self._qcfs_model(theta=2.0)
        raw = model[0].weight.detach().cpu().numpy().copy()
        snn = convert(model, calibration_data=torch.randn(20, 4))
        # Threshold comes from the learned theta, not calibration statistics.
        np.testing.assert_allclose(snn.weights[0], raw / 2.0, rtol=1e-6)

    def test_qcfs_route_pads_missing_output_activation(self) -> None:
        # Two Linear layers but a single QCFS layer -> the trailing weight
        # layer falls back to a unit threshold.
        model = self._qcfs_model()
        snn = convert(model)
        assert snn.n_layers == 2
        assert len(snn.thresholds) == 2

    def test_qcfs_converted_snn_runs(self) -> None:
        snn = convert(self._qcfs_model())
        out = snn.run(np.random.default_rng(3).random((5, 4)))
        assert out.shape == (5, 3)

    def test_membrane_shift_raises_early_firing(self) -> None:
        rng = np.random.default_rng(21)
        weights = [rng.random((6, 4)) * 0.2]
        biases: list[np.ndarray | None] = [None]
        rest = ConvertedSNN(
            weights=weights, biases=biases, thresholds=[1.0], T=8, initial_membrane_fraction=0.0
        )
        shifted = ConvertedSNN(
            weights=weights, biases=biases, thresholds=[1.0], T=8, initial_membrane_fraction=0.5
        )
        x = rng.random((10, 4))
        assert shifted.run(x).sum() >= rest.run(x).sum()

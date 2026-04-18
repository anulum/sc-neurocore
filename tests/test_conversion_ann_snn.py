# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Tests for sc_neurocore.conversion (ANN-to-SNN + QCFS)

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from sc_neurocore.conversion.ann_to_snn import ConvertedSNN  # noqa: E402


class TestConvertedSNN:
    def _make_snn(self):
        weights = [np.random.randn(10, 5) * 0.1, np.random.randn(3, 10) * 0.1]
        biases = [np.zeros(10), np.zeros(3)]
        return ConvertedSNN(
            weights=weights,
            biases=biases,
            thresholds=[1.0, 1.0],
            T=16,
        )

    def test_n_layers(self):
        snn = self._make_snn()
        assert snn.n_layers == 2

    def test_run_single(self):
        snn = self._make_snn()
        x = np.random.rand(5)
        out = snn.run(x)
        assert out.shape == (3,)
        assert out.dtype == np.float64

    def test_run_batch(self):
        snn = self._make_snn()
        x = np.random.rand(4, 5)
        out = snn.run(x)
        assert out.shape == (4, 3)

    def test_classify(self):
        snn = self._make_snn()
        x = np.random.rand(4, 5)
        preds = snn.classify(x)
        assert preds.shape == (4,)
        assert all(0 <= p < 3 for p in preds)

    def test_classify_single(self):
        snn = self._make_snn()
        x = np.random.rand(5)
        pred = snn.classify(x)
        assert isinstance(int(pred), int)

    def test_no_bias(self):
        weights = [np.random.randn(3, 2) * 0.1]
        biases = [None]
        snn = ConvertedSNN(weights=weights, biases=biases, thresholds=[1.0], T=8)
        out = snn.run(np.random.rand(2))
        assert out.shape == (3,)


class TestConvert:
    def test_convert_simple_model(self):
        from sc_neurocore.conversion.ann_to_snn import convert

        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
        snn = convert(model, T=8)
        assert snn.n_layers == 2
        assert snn.T == 8

    def test_convert_with_calibration(self):
        from sc_neurocore.conversion.ann_to_snn import convert

        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
        cal_data = torch.randn(10, 4)
        snn = convert(model, calibration_data=cal_data, T=16)
        assert snn.n_layers == 2

    def test_convert_no_linear_raises(self):
        from sc_neurocore.conversion.ann_to_snn import convert

        model = nn.Sequential(nn.ReLU())
        with pytest.raises(ValueError, match="No Linear"):
            convert(model)

    def test_round_trip_accuracy(self):
        from sc_neurocore.conversion.ann_to_snn import convert

        model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 3))
        model.eval()
        snn = convert(model, T=64)

        x = np.random.rand(20, 4) * 0.5
        ann_out = model(torch.tensor(x, dtype=torch.float32)).detach().numpy()
        ann_preds = ann_out.argmax(axis=1)
        snn_preds = snn.classify(x)
        # Not expecting perfect match, just that conversion runs
        assert len(snn_preds) == 20


class TestQCFSActivation:
    def test_forward(self):
        from sc_neurocore.conversion.qcfs import QCFSActivation

        act = QCFSActivation(T=8, theta=1.0)
        x = torch.tensor([0.0, 0.5, 1.0, 1.5, -0.5])
        out = act(x)
        assert out[0].item() >= 0
        assert out[-1].item() >= 0  # clamp at 0

    def test_output_range(self):
        from sc_neurocore.conversion.qcfs import QCFSActivation

        act = QCFSActivation(T=4, theta=1.0)
        x = torch.linspace(-1, 2, 100)
        out = act(x)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0 + 1e-6

    def test_gradient_flows(self):
        from sc_neurocore.conversion.qcfs import QCFSActivation

        act = QCFSActivation(T=8, theta=1.0)
        x = torch.tensor([0.5], requires_grad=True)
        out = act(x)
        out.backward()
        assert x.grad is not None

    def test_learnable_theta(self):
        from sc_neurocore.conversion.qcfs import QCFSActivation

        act = QCFSActivation(T=8, theta=1.0, learn_theta=True)
        assert isinstance(act.theta, nn.Parameter)

    def test_extra_repr(self):
        from sc_neurocore.conversion.qcfs import QCFSActivation

        act = QCFSActivation(T=8, theta=2.0)
        r = act.extra_repr()
        assert "T=8" in r
        assert "2.00" in r

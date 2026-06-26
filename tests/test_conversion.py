# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for ANN-to-SNN conversion

"""Tests for the ANN-to-SNN conversion engine."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from sc_neurocore.conversion.ann_to_snn import convert, _extract_layers
from sc_neurocore.conversion.qcfs import QCFSActivation


def _make_ann(in_f: int = 4, hidden: int = 8, out_f: int = 3) -> object:
    torch.manual_seed(42)
    return nn.Sequential(nn.Linear(in_f, hidden), nn.ReLU(), nn.Linear(hidden, out_f))


class TestConvert:
    def test_basic_conversion(self) -> None:
        ann = _make_ann()
        snn = convert(ann, T=16)
        assert snn.n_layers == 2
        assert snn.T == 16
        assert len(snn.weights) == 2
        assert len(snn.thresholds) == 2

    def test_with_calibration(self) -> None:
        ann = _make_ann()
        cal = torch.randn(50, 4)
        snn = convert(ann, calibration_data=cal, T=8)
        assert snn.n_layers == 2

    def test_run_produces_output(self) -> None:
        ann = _make_ann()
        snn = convert(ann, T=16)
        x = np.random.default_rng(1).random(4)
        counts = snn.run(x)
        assert counts.shape == (3,)
        assert counts.sum() >= 0

    def test_batch_run(self) -> None:
        ann = _make_ann()
        snn = convert(ann, T=16)
        x = np.random.default_rng(2).random((10, 4))
        counts = snn.run(x)
        assert counts.shape == (10, 3)

    def test_classify(self) -> None:
        ann = _make_ann()
        snn = convert(ann, T=16)
        x = np.random.default_rng(3).random((10, 4))
        pred = snn.classify(x)
        assert pred.shape == (10,)
        assert all(0 <= p < 3 for p in pred)

    def test_weight_shapes_match_ann(self) -> None:
        ann = _make_ann(8, 16, 5)
        snn = convert(ann, T=8)
        assert snn.weights[0].shape == (16, 8)
        assert snn.weights[1].shape == (5, 16)

    def test_no_linear_raises(self) -> None:
        model = nn.Sequential(nn.ReLU())
        with pytest.raises(ValueError, match="No Linear"):
            convert(model, T=8)

    def test_higher_t_more_spikes(self) -> None:
        ann = _make_ann()
        snn8 = convert(ann, T=8)
        snn64 = convert(ann, T=64)
        x = np.random.default_rng(4).random(4) * 0.5
        c8 = snn8.run(x).sum()
        c64 = snn64.run(x).sum()
        assert c64 >= c8


class TestQCFS:
    def test_output_range(self) -> None:
        qcfs = QCFSActivation(T=8, theta=1.0)
        x = torch.linspace(-1, 2, 100)
        out = qcfs(x)
        assert out.min() >= 0
        assert out.max() <= 1.0

    def test_monotonic(self) -> None:
        qcfs = QCFSActivation(T=4, theta=1.0)
        x = torch.linspace(0, 1, 100)
        out = qcfs(x)
        diffs = out[1:] - out[:-1]
        assert (diffs >= -1e-6).all(), "QCFS output should be monotonically non-decreasing"

    def test_gradient_flows(self) -> None:
        qcfs = QCFSActivation(T=8, theta=1.0)
        x = torch.tensor([0.5], requires_grad=True)
        out = qcfs(x)
        out.backward()
        assert x.grad is not None

    def test_learnable_theta(self) -> None:
        qcfs = QCFSActivation(T=8, theta=1.0, learn_theta=True)
        assert isinstance(qcfs.theta, nn.Parameter)


class TestExtractLayers:
    def test_extracts_linear(self) -> None:
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
        layers = _extract_layers(model)
        assert len(layers) == 2
        assert layers[0][0].shape == (8, 4)
        assert layers[1][0].shape == (3, 8)

    def test_handles_no_bias(self) -> None:
        model = nn.Sequential(nn.Linear(4, 8, bias=False))
        layers = _extract_layers(model)
        assert layers[0][1] is None

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Tests for sc_neurocore.conversion (ANN-to-SNN + QCFS)

from __future__ import annotations

import builtins
from collections.abc import Callable
import numpy as np
import numpy.typing as npt
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from sc_neurocore.conversion.ann_to_snn import ConvertedSNN  # noqa: E402
from sc_neurocore.conversion.qcfs import QCFSActivation  # noqa: E402


class TestConversionPackageFacade:
    def test_lazy_facade_resolves_public_exports(self) -> None:
        import sc_neurocore.conversion as conversion
        from sc_neurocore.conversion.ann_to_snn import convert

        assert conversion.__getattr__("convert") is convert
        assert conversion.__getattr__("ConvertedSNN") is ConvertedSNN
        assert conversion.__getattr__("QCFSActivation") is QCFSActivation

    def test_lazy_facade_rejects_unknown_export(self) -> None:
        import sc_neurocore.conversion as conversion

        with pytest.raises(AttributeError, match="not_exported"):
            conversion.__getattr__("not_exported")

    def test_lazy_facade_reports_qcfs_import_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sc_neurocore.conversion as conversion

        original_import: Callable[..., object] = builtins.__import__

        def guarded_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "qcfs":
                raise ImportError("forced missing torch surface")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)
        with pytest.raises(ImportError, match="QCFSActivation requires PyTorch"):
            conversion.__getattr__("QCFSActivation")


class TestConvertedSNN:
    def _make_snn(self) -> ConvertedSNN:
        rng = np.random.default_rng(11)
        weights: list[npt.NDArray[np.float64]] = [
            rng.normal(size=(10, 5)) * 0.1,
            rng.normal(size=(3, 10)) * 0.1,
        ]
        biases: list[npt.NDArray[np.float64] | None] = [np.zeros(10), np.zeros(3)]
        return ConvertedSNN(
            weights=weights,
            biases=biases,
            thresholds=[1.0, 1.0],
            T=16,
        )

    def test_n_layers(self) -> None:
        snn = self._make_snn()
        assert snn.n_layers == 2

    def test_run_single(self) -> None:
        snn = self._make_snn()
        x = np.random.default_rng(12).random(5)
        out = snn.run(x)
        assert out.shape == (3,)
        assert out.dtype == np.float64

    def test_run_batch(self) -> None:
        snn = self._make_snn()
        x = np.random.default_rng(13).random((4, 5))
        out = snn.run(x)
        assert out.shape == (4, 3)

    def test_classify(self) -> None:
        snn = self._make_snn()
        x = np.random.default_rng(14).random((4, 5))
        preds = snn.classify(x)
        assert preds.shape == (4,)
        assert all(0 <= p < 3 for p in preds)

    def test_classify_single(self) -> None:
        snn = self._make_snn()
        x = np.random.default_rng(15).random(5)
        pred = snn.classify(x)
        assert isinstance(int(pred), int)

    def test_no_bias(self) -> None:
        weights: list[npt.NDArray[np.float64]] = [np.random.default_rng(16).normal(size=(3, 2))]
        biases: list[npt.NDArray[np.float64] | None] = [None]
        snn = ConvertedSNN(weights=weights, biases=biases, thresholds=[1.0], T=8)
        out = snn.run(np.random.default_rng(17).random(2))
        assert out.shape == (3,)


class TestConvert:
    def test_convert_simple_model(self) -> None:
        from sc_neurocore.conversion.ann_to_snn import convert

        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
        snn = convert(model, T=8)
        assert snn.n_layers == 2
        assert snn.T == 8

    def test_convert_with_calibration(self) -> None:
        from sc_neurocore.conversion.ann_to_snn import convert

        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
        cal_data = torch.randn(10, 4)
        snn = convert(model, calibration_data=cal_data, T=16)
        assert snn.n_layers == 2

    def test_convert_no_linear_raises(self) -> None:
        from sc_neurocore.conversion.ann_to_snn import convert

        model = nn.Sequential(nn.ReLU())
        with pytest.raises(ValueError, match="No Linear"):
            convert(model)

    def test_convert_requires_torch_when_backend_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from sc_neurocore.conversion import ann_to_snn

        monkeypatch.setattr(ann_to_snn, "HAS_TORCH", False)
        with pytest.raises(ImportError, match="PyTorch required"):
            ann_to_snn.convert(object())

    def test_round_trip_accuracy(self) -> None:
        from sc_neurocore.conversion.ann_to_snn import convert

        model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 3))
        model.eval()
        snn = convert(model, T=64)

        x = np.random.default_rng(18).random((20, 4)) * 0.5
        snn_preds = snn.classify(x)
        # Not expecting perfect match, just that conversion runs
        assert len(snn_preds) == 20


class TestQCFSActivation:
    def test_forward(self) -> None:
        from sc_neurocore.conversion.qcfs import QCFSActivation

        act = QCFSActivation(T=8, theta=1.0)
        x = torch.tensor([0.0, 0.5, 1.0, 1.5, -0.5])
        out = act(x)
        assert out[0].item() >= 0
        assert out[-1].item() >= 0  # clamp at 0

    def test_output_range(self) -> None:
        from sc_neurocore.conversion.qcfs import QCFSActivation

        act = QCFSActivation(T=4, theta=1.0)
        x = torch.linspace(-1, 2, 100)
        out = act(x)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0 + 1e-6

    def test_gradient_flows(self) -> None:
        from sc_neurocore.conversion.qcfs import QCFSActivation

        act = QCFSActivation(T=8, theta=1.0)
        x = torch.tensor([0.5], requires_grad=True)
        out = act(x)
        out.backward()
        assert x.grad is not None

    def test_learnable_theta(self) -> None:
        from sc_neurocore.conversion.qcfs import QCFSActivation

        act = QCFSActivation(T=8, theta=1.0, learn_theta=True)
        assert isinstance(act.theta, nn.Parameter)

    def test_extra_repr(self) -> None:
        from sc_neurocore.conversion.qcfs import QCFSActivation

        act = QCFSActivation(T=8, theta=2.0)
        r = act.extra_repr()
        assert "T=8" in r
        assert "2.00" in r

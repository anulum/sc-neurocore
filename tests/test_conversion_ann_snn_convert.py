# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestConvert from former test_conversion_ann_snn.py

"""Focused suite: TestConvert from former test_conversion_ann_snn.py."""

from __future__ import annotations

from tests.conversion_ann_snn_support import *  # noqa: F403

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

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestConvert from former test_conversion.py

"""Focused suite: TestConvert from former test_conversion.py."""

from __future__ import annotations

import numpy as np
import pytest
from tests.conversion_support import (
    _make_ann,
    convert,
    nn,
    torch,
)


class TestConvert:
    """Converting a small ANN to an SNN."""

    def test_basic_conversion(self) -> None:
        """Conversion gives two layers and 16 default time steps."""
        ann = _make_ann()
        snn = convert(ann, T=16)
        assert snn.n_layers == 2
        assert snn.T == 16
        assert len(snn.weights) == 2
        assert len(snn.thresholds) == 2

    def test_with_calibration(self) -> None:
        """Conversion with calibration data keeps the layer count."""
        ann = _make_ann()
        cal = torch.randn(50, 4)
        snn = convert(ann, calibration_data=cal, T=8)
        assert snn.n_layers == 2

    def test_run_produces_output(self) -> None:
        """A run gives one count per output."""
        ann = _make_ann()
        snn = convert(ann, T=16)
        x = np.random.default_rng(1).random(4)
        counts = snn.run(x)
        assert counts.shape == (3,)
        assert counts.sum() >= 0

    def test_batch_run(self) -> None:
        """A batch run gives one row per sample."""
        ann = _make_ann()
        snn = convert(ann, T=16)
        x = np.random.default_rng(2).random((10, 4))
        counts = snn.run(x)
        assert counts.shape == (10, 3)

    def test_classify(self) -> None:
        """Classification gives valid class indices."""
        ann = _make_ann()
        snn = convert(ann, T=16)
        x = np.random.default_rng(3).random((10, 4))
        pred = snn.classify(x)
        assert pred.shape == (10,)
        assert all(0 <= p < 3 for p in pred)

    def test_weight_shapes_match_ann(self) -> None:
        """The converted weights keep the ANN shapes."""
        ann = _make_ann(8, 16, 5)
        snn = convert(ann, T=8)
        assert snn.weights[0].shape == (16, 8)
        assert snn.weights[1].shape == (5, 16)

    def test_no_linear_raises(self) -> None:
        """A model without Linear layers is rejected."""
        model = nn.Sequential(nn.ReLU())
        with pytest.raises(ValueError, match="No Linear"):
            convert(model, T=8)

    def test_higher_t_more_spikes(self) -> None:
        """More time steps give no fewer spikes."""
        ann = _make_ann()
        snn8 = convert(ann, T=8)
        snn64 = convert(ann, T=64)
        x = np.random.default_rng(4).random(4) * 0.5
        c8 = snn8.run(x).sum()
        c64 = snn64.run(x).sum()
        assert c64 >= c8

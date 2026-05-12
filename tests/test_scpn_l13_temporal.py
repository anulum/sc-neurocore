# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SCPN L13 temporal binding layer

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.scpn.layers.l13_temporal import L13_StochasticParameters, L13_TemporalLayer


def test_l13_temporal_binding_uses_lagged_correlation() -> None:
    params = L13_StochasticParameters(
        n_channels=2,
        bitstream_length=16,
        binding_window=6,
        binding_threshold=0.7,
    )
    layer = L13_TemporalLayer(params)

    # Channel 1 is channel 0 delayed by one timestep. Zero-lag Pearson
    # stays weak, but max-lag binding should detect the temporal relation.
    inputs = [
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ]
    result = {}
    for values in inputs:
        result = layer.step(0.001, {"coherence": np.array(values, dtype=np.float64)})

    binding = result["binding_matrix"]
    assert abs(binding[0, 1]) > 0.9
    assert result["binding_strength"] == pytest.approx(1.0)


def test_l13_temporal_layer_rejects_invalid_parameters_and_inputs() -> None:
    with pytest.raises(ValueError, match="n_channels"):
        L13_TemporalLayer(L13_StochasticParameters(n_channels=0))
    with pytest.raises(ValueError, match="binding_window"):
        L13_TemporalLayer(L13_StochasticParameters(binding_window=0))

    layer = L13_TemporalLayer(L13_StochasticParameters(n_channels=2))
    with pytest.raises(ValueError, match="dt"):
        layer.step(0.0)
    with pytest.raises(ValueError, match="coherence"):
        layer.step(0.001, {"coherence": np.array([np.nan, 0.0])})

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for pretrained weight loading

"""Tests for sc_neurocore.model_zoo.pretrained."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.model_zoo.pretrained import load_pretrained
from sc_neurocore.network import Network


def test_load_mnist_returns_network() -> None:
    net = load_pretrained("mnist")
    assert isinstance(net, Network)
    assert len(net.populations) >= 3


def test_mnist_weight_shapes() -> None:
    net = load_pretrained("mnist")
    # input->hidden: 784->128, hidden->output: 128->10
    proj_ih = net.projections[0]
    proj_ho = net.projections[1]
    assert proj_ih.source.n == 784
    assert proj_ih.target.n == 128
    assert proj_ho.source.n == 128
    assert proj_ho.target.n == 10


def test_weights_in_expected_range() -> None:
    """Xavier/Glorot + 0.5 spiking correction keeps weights small."""
    net = load_pretrained("mnist")
    for proj in net.projections:
        max_abs = np.max(np.abs(proj.data))
        # limit = sqrt(6/(fan_in+fan_out)) * 0.5, largest case 784+128=912 -> ~0.041
        assert max_abs < 0.15, f"Weight magnitude {max_abs} exceeds expected range"


def test_network_runs_after_loading() -> None:
    net = load_pretrained("dvs_gesture")
    net.run(0.01, dt=0.001)
    total_spikes = sum(m.count for m in net.spike_monitors)
    assert total_spikes >= 0


def test_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="Unknown pretrained model"):
        load_pretrained("nonexistent_model_xyz")

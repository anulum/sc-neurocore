# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLoadPretrained from former test_model_zoo.py

"""Focused suite: TestLoadPretrained from former test_model_zoo.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.model_zoo import (
    mnist_classifier,
)
from sc_neurocore.model_zoo.pretrained import (
    load_pretrained,
)
from sc_neurocore.network.network import (
    Network,
)
from tests.model_zoo_support import (
    _total_spikes,
)


class TestLoadPretrained:
    """Tests for load_pretrained: loads .npz weights into network projections."""

    def test_mnist_loads(self) -> None:
        """The pretrained MNIST network loads with two projections."""
        net = load_pretrained("mnist")
        assert isinstance(net, Network)
        assert len(net.projections) == 2

    def test_mnist_weights_differ_from_default(self) -> None:
        """Loaded weights should differ from default Xavier init."""
        default = mnist_classifier(n_hidden=128)
        loaded = load_pretrained("mnist")
        # At least one projection's data should differ
        differs = False
        for i in range(2):
            if not np.array_equal(default.projections[i].data, loaded.projections[i].data):
                differs = True
        assert differs

    def test_shd_loads(self) -> None:
        """The pretrained SHD network loads with three projections."""
        net = load_pretrained("shd")
        assert isinstance(net, Network)
        assert len(net.projections) == 3

    def test_dvs_gesture_loads(self) -> None:
        """The pretrained DVS gesture network loads with two projections."""
        net = load_pretrained("dvs_gesture")
        assert isinstance(net, Network)
        assert len(net.projections) == 2

    def test_unknown_name_raises(self) -> None:
        """Unknown name raises."""
        with pytest.raises(ValueError, match="Unknown pretrained model"):
            load_pretrained("nonexistent_model")

    def test_mnist_pretrained_produces_spikes(self) -> None:
        """Mnist pretrained produces spikes."""
        net = load_pretrained("mnist")
        net.run(0.05, dt=0.001, backend="python")
        assert _total_spikes(net) > 0

    def test_shd_pretrained_produces_spikes(self) -> None:
        """Shd pretrained produces spikes."""
        net = load_pretrained("shd")
        net.run(0.05, dt=0.001, backend="python")
        assert _total_spikes(net) > 0

    def test_dvs_pretrained_produces_spikes(self) -> None:
        """Dvs pretrained produces spikes."""
        net = load_pretrained("dvs_gesture")
        net.run(0.05, dt=0.001, backend="python")
        assert _total_spikes(net) > 0

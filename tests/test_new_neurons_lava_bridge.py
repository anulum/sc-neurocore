# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLavaBridge from former test_new_neurons.py

"""Focused suite: TestLavaBridge from former test_new_neurons.py."""

from __future__ import annotations

from tests.new_neurons_support import *  # noqa: F403

class TestLavaBridge:
    def test_export_weights(self):
        import numpy as np
        from sc_neurocore.integrations.lava_bridge import export_weights_loihi

        w = np.array([[0.0, 0.5, 1.0], [0.25, 0.75, 0.5]])
        loihi_w = export_weights_loihi(w, weight_bits=8)
        assert loihi_w.dtype == np.int32
        assert loihi_w.shape == (2, 3)
        assert loihi_w[0, 0] == -127  # 0.0 → -1.0 → -127
        assert loihi_w[0, 2] == 127  # 1.0 → +1.0 → +127

    def test_converter(self):
        from sc_neurocore.integrations.lava_bridge import SCtoLavaConverter, LoihiNetworkConfig

        converter = SCtoLavaConverter(weight_bits=8)

        class FakeLayer:
            weights = [[0.5, 0.3], [0.7, 0.1], [0.9, 0.4]]

        config = converter.convert_dense_layer(FakeLayer())
        assert isinstance(config, LoihiNetworkConfig)
        assert config.n_inputs == 2
        assert config.n_outputs == 3

    def test_threshold_conversion(self):
        from sc_neurocore.integrations.lava_bridge import loihi_threshold_from_sc

        t = loihi_threshold_from_sc(1.0, weight_bits=8)
        assert t == 127

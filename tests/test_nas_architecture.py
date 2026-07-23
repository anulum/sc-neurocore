# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestArchitecture from former test_nas.py

"""Focused suite: TestArchitecture from former test_nas.py."""

from __future__ import annotations

from tests.nas_support import *  # noqa: F403

class TestArchitecture:
    def test_fields(self) -> None:
        a = Architecture(
            n_inputs=64,
            layer_widths=[32, 16],
            neuron_types=["StochasticLIFNeuron", "StochasticLIFNeuron"],
            bitstream_lengths=[128, 64],
            delay_ranges=[2, 0],
        )
        assert a.n_layers == 2
        assert a.layer_sizes == [(64, 32), (32, 16)]
        assert a.total_params == 64 * 32 + 32 * 16

    def test_single_layer(self) -> None:
        a = Architecture(
            n_inputs=10,
            layer_widths=[5],
            neuron_types=["StochasticLIFNeuron"],
            bitstream_lengths=[256],
            delay_ranges=[0],
        )
        assert a.n_layers == 1
        assert a.total_params == 50

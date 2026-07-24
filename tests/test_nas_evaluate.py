# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEvaluate from former test_nas.py

"""Focused suite: TestEvaluate from former test_nas.py."""

from __future__ import annotations

from tests.nas_support import *  # noqa: F403


class TestEvaluate:
    def test_default_proxy(self) -> None:
        arch = Architecture(
            n_inputs=64,
            layer_widths=[32, 10],
            neuron_types=["StochasticLIFNeuron"] * 2,
            bitstream_lengths=[128, 128],
            delay_ranges=[0, 0],
        )
        _evaluate(arch, "artix7")
        assert arch.fitness_luts > 0
        assert arch.fitness_energy_nj > 0
        assert 0 < arch.fitness_accuracy <= 1.0

    def test_custom_accuracy_fn(self) -> None:
        arch = Architecture(
            n_inputs=16,
            layer_widths=[8],
            neuron_types=["StochasticLIFNeuron"],
            bitstream_lengths=[64],
            delay_ranges=[0],
        )
        _evaluate(arch, "ice40", accuracy_fn=lambda a: 0.95)
        assert arch.fitness_accuracy == 0.95

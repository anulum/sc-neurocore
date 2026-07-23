# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestChipSpec from former test_chip_compiler.py

"""Focused suite: TestChipSpec from former test_chip_compiler.py."""

from __future__ import annotations

from tests.chip_compiler_support import *  # noqa: F403

class TestChipSpec:
    def test_loihi2(self):
        chip = BUILTIN_CHIPS["loihi2"]
        assert chip.total_neurons == 128 * 128
        assert chip.core.has_on_chip_learning
        assert "LIF" in chip.core.supported_neuron_types

    def test_xylo(self):
        chip = BUILTIN_CHIPS["xylo"]
        assert chip.total_neurons == 1000
        assert not chip.core.has_on_chip_learning

    def test_speck(self):
        chip = BUILTIN_CHIPS["speck"]
        assert chip.core.weight_bits == 4

    def test_akida(self):
        chip = BUILTIN_CHIPS["akida"]
        assert chip.vendor == "BrainChip"

    def test_spinnaker2(self):
        chip = BUILTIN_CHIPS["spinnaker2"]
        assert chip.core.max_delay_steps == 256

    def test_brainscales2(self):
        chip = BUILTIN_CHIPS["brainscales2"]
        assert chip.analog_noise_cv == 0.20

    def test_fits(self):
        chip = BUILTIN_CHIPS["loihi2"]
        assert chip.fits(1000)
        assert not chip.fits(100000)

    def test_cores_needed(self):
        chip = BUILTIN_CHIPS["loihi2"]
        assert chip.cores_needed(100) == 1
        assert chip.cores_needed(200) == 2
        assert chip.cores_needed(128) == 1

    def test_total_power(self):
        chip = BUILTIN_CHIPS["loihi2"]
        assert chip.total_power_mw == 128 * 0.5

    def test_builtin_count(self):
        assert len(BUILTIN_CHIPS) >= 6

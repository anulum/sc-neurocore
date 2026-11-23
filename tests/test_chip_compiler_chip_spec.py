# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestChipSpec from former test_chip_compiler.py

"""Focused suite: TestChipSpec from former test_chip_compiler.py."""

from __future__ import annotations

from sc_neurocore.chip_compiler import (
    BUILTIN_CHIPS,
)


class TestChipSpec:
    """The built-in neuromorphic chip specifications."""

    def test_loihi2(self) -> None:
        """Loihi 2 has 128 by 128 neurons and on-chip learning."""
        chip = BUILTIN_CHIPS["loihi2"]
        assert chip.total_neurons == 128 * 128
        assert chip.core.has_on_chip_learning
        assert "LIF" in chip.core.supported_neuron_types

    def test_xylo(self) -> None:
        """Xylo has 1000 neurons and no on-chip learning."""
        chip = BUILTIN_CHIPS["xylo"]
        assert chip.total_neurons == 1000
        assert not chip.core.has_on_chip_learning

    def test_speck(self) -> None:
        """Speck uses 4-bit weights."""
        chip = BUILTIN_CHIPS["speck"]
        assert chip.core.weight_bits == 4

    def test_akida(self) -> None:
        """Akida is made by BrainChip."""
        chip = BUILTIN_CHIPS["akida"]
        assert chip.vendor == "BrainChip"

    def test_spinnaker2(self) -> None:
        """SpiNNaker 2 allows 256 delay steps."""
        chip = BUILTIN_CHIPS["spinnaker2"]
        assert chip.core.max_delay_steps == 256

    def test_brainscales2(self) -> None:
        """BrainScaleS-2 has an analogue noise CV of 0.20."""
        chip = BUILTIN_CHIPS["brainscales2"]
        assert chip.analog_noise_cv == 0.20

    def test_fits(self) -> None:
        """A chip fits a small network and not a huge one."""
        chip = BUILTIN_CHIPS["loihi2"]
        assert chip.fits(1000)
        assert not chip.fits(100000)

    def test_cores_needed(self) -> None:
        """The core count rounds the neurons up per core."""
        chip = BUILTIN_CHIPS["loihi2"]
        assert chip.cores_needed(100) == 1
        assert chip.cores_needed(200) == 2
        assert chip.cores_needed(128) == 1

    def test_total_power(self) -> None:
        """The total power is the per-core power times the cores."""
        chip = BUILTIN_CHIPS["loihi2"]
        assert chip.total_power_mw == 128 * 0.5

    def test_builtin_count(self) -> None:
        """At least six chips are built in."""
        assert len(BUILTIN_CHIPS) >= 6

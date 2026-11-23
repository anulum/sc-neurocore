# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMLCConfig from former test_spintronic_mapper.py

"""Focused suite: TestMLCConfig from former test_spintronic_mapper.py."""

from __future__ import annotations

from sc_neurocore.spintronic.spintronic_mapper import (
    MLCConfig,
)


class TestMLCConfig:
    """Multi-level cell quantisation of weights."""

    def test_levels(self) -> None:
        """Two bits per cell give four levels."""
        mlc = MLCConfig(bits_per_cell=2)
        assert mlc.levels == 4

    def test_quantize(self) -> None:
        """Weights 0 and 1 map to the lowest and highest level."""
        mlc = MLCConfig(bits_per_cell=2)
        assert mlc.quantize_weight(0.0) == 0
        assert mlc.quantize_weight(1.0) == 3

    def test_dequantize(self) -> None:
        """The lowest and highest level map back to 0 and 1."""
        mlc = MLCConfig(bits_per_cell=2)
        assert mlc.dequantize(0) == 0.0
        assert abs(mlc.dequantize(3) - 1.0) < 0.01

    def test_density(self) -> None:
        """Three bits per cell triple the density."""
        assert MLCConfig(bits_per_cell=3).density_improvement == 3.0

    def test_resistance_margins_span_parallel_to_antiparallel(self) -> None:
        """The four level resistances run from parallel to antiparallel."""
        margins = MLCConfig(bits_per_cell=2).resistance_margins
        assert len(margins) == 4
        assert margins[0] == 5000.0
        assert margins[-1] == 12500.0

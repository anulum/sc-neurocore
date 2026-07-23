# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFoldedHeterogeneousParams from former test_folded_heterogeneous_params.py

"""Focused suite: TestFoldedHeterogeneousParams from former test_folded_heterogeneous_params.py."""

from __future__ import annotations

from tests.folded_heterogeneous_params_support import *  # noqa: F403

class TestFoldedHeterogeneousParams:
    """The folded interconnect streams per-neuron parameters through a ROM."""

    def test_heterogeneous_population_folds(self) -> None:
        result = _compile([10.0, 20.0, 30.0], "folded")
        assert result.interconnect == "folded"

    def test_folded_pe_exposes_the_varying_parameter_as_a_port(self) -> None:
        result = _compile([10.0, 20.0, 30.0], "folded")
        assert any("input wire signed [15:0] P_TAU" in m for m in result.neuron_modules.values())
        assert ".P_TAU(param_tau_lif)" in result.top_module

    def test_folded_emits_a_per_neuron_parameter_rom(self) -> None:
        result = _compile([10.0, 20.0, 30.0], "folded")
        top = result.top_module
        # The ROM carries each neuron's own quantised tau (2560/5120/7680), addressed by nidx.
        assert f"= {16}'sd{_Q.encode(20.0)}" in top
        assert f"= {16}'sd{_Q.encode(30.0)}" in top
        assert "case (nidx)" in top

    def test_folded_reports_parameter_rom_bits(self) -> None:
        result = _compile([10.0, 20.0, 30.0], "folded")
        assert result.folded_metrics is not None
        # 3 neurons × 1 varying parameter × 16 bits.
        assert result.folded_metrics.param_rom_bits == 3 * 1 * 16
        assert result.folded_metrics.as_dict()["param_rom_bits"] == 48

    def test_homogeneous_population_has_no_parameter_rom(self) -> None:
        result = _compile([10.0, 10.0, 10.0], "folded")
        assert "param_tau" not in result.top_module
        assert result.folded_metrics is not None
        assert result.folded_metrics.param_rom_bits == 0

    def test_folded_pe_bakes_the_real_parameter_not_a_double_encoded_zero(self) -> None:
        # Regression: the folded PE built the per-type neuron from the *quantised* population,
        # so Q88.encode ran twice and baked tau = 5120 × 256 mod 2**16 = 0 into the shared PE
        # for every real graph. A uniform explicit tau must bake q.encode(20) = 5120, not 0.
        result = _compile([20.0, 20.0, 20.0], "folded")
        pe = next(src for key, src in result.neuron_modules.items() if key.endswith("_pe"))
        assert f"P_TAU = 16'sd{_Q.encode(20.0)}" in pe
        assert "P_TAU = 16'sd0" not in pe

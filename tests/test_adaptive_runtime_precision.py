# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — E2E tests for adaptive runtime precision

"""End-to-end tests for adaptive precision telemetry (Roadmap Item 6).

Tests verify:
- Dual-datapath generation (LP + HP sub-modules + wrapper)
- HP-authoritative outputs with no fabric clock gating
- Hysteresis thresholds (THRESH_UP, THRESH_DOWN)
- LP datapath remains available for telemetry
- All canonical Q-format LP/HP pairs
- Validation of invalid configurations
- Structural completeness of generated Verilog
"""

from __future__ import annotations

import pytest

from sc_neurocore.compiler.adaptive_runtime_precision import (
    PRECISION_PAIRS,
    compile_adaptive_precision,
)
from sc_neurocore.neurons.equation_builder import from_equations

import json


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def lif_neuron():
    """Standard LIF neuron for adaptive precision tests."""
    return from_equations(
        "dv/dt = -(v - E_L)/tau_m + I/C",
        threshold="v > -50",
        reset="v = -65",
        params=dict(E_L=-65, tau_m=10, C=1),
        init=dict(v=-65),
    )


@pytest.fixture
def izhikevich_neuron():
    """Two-state-variable neuron (Izhikevich)."""
    return from_equations(
        "dv/dt = 0.04 * v * v + 5 * v + 140 - u + I",
        "du/dt = a * (b * v - u)",
        threshold="v > 30",
        reset="v = c; u = u + d",
        params=dict(a=0.02, b=0.2, c=-65, d=8),
        init=dict(v=-65, u=-14),
    )


# ═══════════════════════════════════════════════════════════════════════
# 1. Dual-datapath generation
# ═══════════════════════════════════════════════════════════════════════


class TestDualDatapath:
    """Verify that both LP and HP datapaths are generated."""

    def test_contains_lp_module(self, lif_neuron):
        """LP sub-module must be present."""
        v = compile_adaptive_precision(lif_neuron, module_name="sc_lif_adapt")
        assert "module sc_lif_adapt_lp" in v

    def test_contains_hp_module(self, lif_neuron):
        """HP sub-module must be present."""
        v = compile_adaptive_precision(lif_neuron, module_name="sc_lif_adapt")
        assert "module sc_lif_adapt_hp" in v

    def test_contains_wrapper_module(self, lif_neuron):
        """Top-level wrapper module must be present."""
        v = compile_adaptive_precision(lif_neuron, module_name="sc_lif_adapt")
        assert "module sc_lif_adapt " in v or "module sc_lif_adapt\n" in v

    def test_lp_instantiation(self, lif_neuron):
        """LP datapath must be instantiated."""
        v = compile_adaptive_precision(lif_neuron, module_name="sc_lif_adapt")
        assert "lp_inst" in v

    def test_hp_instantiation(self, lif_neuron):
        """HP datapath must be instantiated."""
        v = compile_adaptive_precision(lif_neuron, module_name="sc_lif_adapt")
        assert "hp_inst" in v

    def test_three_endmodule(self, lif_neuron):
        """Must have 3 endmodule statements (LP, HP, wrapper)."""
        v = compile_adaptive_precision(lif_neuron, module_name="sc_lif_adapt")
        assert v.count("endmodule") == 3


# ═══════════════════════════════════════════════════════════════════════
# 2. HP-authoritative clocking
# ═══════════════════════════════════════════════════════════════════════


class TestHPAuthoritativeClocking:
    """Verify the HP datapath remains clocked and authoritative."""

    def test_no_hp_clock_gate(self, lif_neuron):
        """Generated RTL must not gate clk with use_hp in fabric."""
        v = compile_adaptive_precision(lif_neuron)
        assert "hp_clk" not in v

    def test_no_clk_and_use_hp(self, lif_neuron):
        """Generated RTL must not create clk & use_hp."""
        v = compile_adaptive_precision(lif_neuron)
        assert "clk & use_hp" not in v

    def test_hp_inst_uses_clk(self, lif_neuron):
        """HP instance must use the primary clock."""
        v = compile_adaptive_precision(lif_neuron)
        hp_inst = v.split("hp_inst", 1)[1]
        assert ".clk(clk)" in hp_inst

    def test_use_hp_port(self, lif_neuron):
        """use_hp output port must be present as telemetry."""
        v = compile_adaptive_precision(lif_neuron)
        assert "output wire use_hp" in v


# ═══════════════════════════════════════════════════════════════════════
# 3. Hysteresis thresholds
# ═══════════════════════════════════════════════════════════════════════


class TestHysteresis:
    """Verify hysteresis threshold logic."""

    def test_thresh_up_present(self, lif_neuron):
        """THRESH_UP localparam must be present."""
        v = compile_adaptive_precision(lif_neuron)
        assert "THRESH_UP" in v

    def test_thresh_down_present(self, lif_neuron):
        """THRESH_DOWN localparam must be present."""
        v = compile_adaptive_precision(lif_neuron)
        assert "THRESH_DOWN" in v

    def test_precision_mode_register(self, lif_neuron):
        """precision_mode register must be declared."""
        v = compile_adaptive_precision(lif_neuron)
        assert "reg precision_mode" in v

    def test_precision_mode_reset_to_lp(self, lif_neuron):
        """precision_mode must reset to 0 (LP mode)."""
        v = compile_adaptive_precision(lif_neuron)
        assert "precision_mode <= 1'b0" in v

    def test_custom_hysteresis(self, lif_neuron):
        """Custom hysteresis percentages should work."""
        v = compile_adaptive_precision(
            lif_neuron,
            threshold_up_pct=0.9,
            threshold_down_pct=0.3,
        )
        assert "90%" in v
        assert "30%" in v


# ═══════════════════════════════════════════════════════════════════════
# 4. HP-authoritative output path
# ═══════════════════════════════════════════════════════════════════════


class TestHPAuthoritativeOutput:
    """Verify outputs are taken from HP, never LP-converted state."""

    def test_output_register_uses_hp_spike(self, lif_neuron):
        """Spike output must be driven by HP."""
        v = compile_adaptive_precision(lif_neuron, lp_width=16, lp_frac=8, hp_width=32, hp_frac=16)
        assert "spike_out <= hp_spike;" in v

    def test_output_register_uses_hp_state(self, lif_neuron):
        """State output must be driven by HP."""
        v = compile_adaptive_precision(lif_neuron, lp_width=16, lp_frac=8, hp_width=32, hp_frac=16)
        assert "v_out <= hp_v_out;" in v
        assert "v_out <= lp_v_out" not in v


# ═══════════════════════════════════════════════════════════════════════
# 5. All canonical LP/HP precision pairs
# ═══════════════════════════════════════════════════════════════════════


class TestAllPrecisionPairs:
    """Verify all canonical LP/HP pairs from PRECISION_PAIRS."""

    @pytest.mark.parametrize(
        "lp_hp",
        PRECISION_PAIRS,
        ids=[
            f"Q{lp[0] - lp[1] - 1}.{lp[1]}_to_Q{hp[0] - hp[1] - 1}.{hp[1]}"
            for lp, hp in PRECISION_PAIRS
        ],
    )
    def test_canonical_pair_compiles(self, lif_neuron, lp_hp):
        """Each canonical LP/HP pair must compile without error."""
        (lp_w, lp_f), (hp_w, hp_f) = lp_hp
        v = compile_adaptive_precision(
            lif_neuron,
            module_name="sc_lif_test",
            lp_width=lp_w,
            lp_frac=lp_f,
            hp_width=hp_w,
            hp_frac=hp_f,
        )
        assert "module sc_lif_test_lp" in v
        assert "module sc_lif_test_hp" in v
        assert v.count("endmodule") == 3


def _extract_manifest(verilog: str) -> dict:
    """Extract adaptive precision manifest JSON from generated RTL comments."""
    prefix = "// SC-NeuroCore Adaptive Precision Manifest: "
    for line in verilog.splitlines():
        if line.startswith(prefix):
            return json.loads(line[len(prefix) :])
    raise AssertionError("Adaptive precision manifest comment not found")


class TestPrecisionStrings:
    """Verify new precision-string support and manifest emission."""

    def test_precision_string_api(self, lif_neuron):
        """Q-format strings should be resolved and emitted deterministically."""
        v = compile_adaptive_precision(
            lif_neuron,
            module_name="sc_lif_adapt_precision_strings",
            lp_precision="Q8.8",
            hp_precision="Q16.16",
        )
        manifest = _extract_manifest(v)
        assert manifest["lp_precision"]["kind"] == "fixed"
        assert manifest["hp_precision"]["kind"] == "fixed"
        assert manifest["lp_precision"]["label"] == "Q8.8"
        assert manifest["hp_precision"]["label"] == "Q16.16"

    def test_block_floating_precision_metadata(self, lif_neuron):
        """Block-floating precision should emit block metadata and deterministic label."""
        v = compile_adaptive_precision(
            lif_neuron,
            module_name="sc_lif_adapt_bfp",
            lp_precision="BFP16E3X32",
            hp_precision="Q16.16",
        )
        manifest = _extract_manifest(v)
        assert manifest["lp_precision"]["kind"] == "block_floating"
        assert manifest["lp_precision"]["mantissa_bits"] == 16
        assert manifest["lp_precision"]["exponent_bits"] == 3
        assert manifest["lp_precision"]["block_size"] == 32
        assert manifest["lp_precision"]["label"].startswith("BFP16E3")

    def test_invalid_precision_string(self, lif_neuron):
        """Invalid precision strings must fail with ValueError."""
        with pytest.raises(ValueError, match="precision"):
            compile_adaptive_precision(
                lif_neuron,
                lp_precision="definitely-not-a-format",
                hp_precision="Q16.16",
            )


# ═══════════════════════════════════════════════════════════════════════
# 6. Validation
# ═══════════════════════════════════════════════════════════════════════


class TestValidation:
    """Verify that invalid configurations are rejected."""

    def test_thresholds_require_ordered_band(self, lif_neuron):
        """Swapped hysteresis thresholds must be rejected."""
        with pytest.raises(ValueError, match="threshold_down_pct"):
            compile_adaptive_precision(
                lif_neuron,
                threshold_up_pct=0.2,
                threshold_down_pct=0.8,
            )

        with pytest.raises(ValueError, match="threshold_up_pct"):
            compile_adaptive_precision(
                lif_neuron,
                threshold_up_pct=1.0,
                threshold_down_pct=0.2,
            )

    def test_thresholds_reject_nonfinite(self, lif_neuron):
        """NaN and infinities are rejected by threshold validation."""
        with pytest.raises(ValueError, match="finite"):
            compile_adaptive_precision(
                lif_neuron,
                threshold_up_pct=float("nan"),
                threshold_down_pct=0.2,
            )

        with pytest.raises(ValueError, match="finite"):
            compile_adaptive_precision(
                lif_neuron,
                threshold_up_pct=0.9,
                threshold_down_pct=float("inf"),
            )

        with pytest.raises(ValueError, match="must satisfy 0 < threshold_down_pct"):
            compile_adaptive_precision(
                lif_neuron,
                threshold_up_pct=0.6,
                threshold_down_pct=0.0,
            )

        with pytest.raises(ValueError, match="Quantised threshold codes"):
            compile_adaptive_precision(
                lif_neuron,
                threshold_up_pct=0.00001,
                threshold_down_pct=0.000001,
            )

    def test_thresholds_are_reflected_in_manifest(self, lif_neuron):
        """Manifest must retain threshold policy under compiler contract."""
        up = 0.9
        down = 0.3
        v = compile_adaptive_precision(lif_neuron, threshold_up_pct=up, threshold_down_pct=down)
        manifest = _extract_manifest(v)

        assert manifest["threshold_up_pct"] == up
        assert manifest["threshold_down_pct"] == down

    def test_lp_wider_than_hp_rejected(self, lif_neuron):
        """LP wider than HP must raise ValueError."""
        with pytest.raises(ValueError, match="strictly less"):
            compile_adaptive_precision(lif_neuron, lp_width=32, lp_frac=16, hp_width=16, hp_frac=8)

    def test_equal_widths_rejected(self, lif_neuron):
        """Equal LP and HP widths must raise ValueError."""
        with pytest.raises(ValueError, match="strictly less"):
            compile_adaptive_precision(lif_neuron, lp_width=16, lp_frac=8, hp_width=16, hp_frac=8)

    def test_zero_frac_rejected(self, lif_neuron):
        """Zero fractional bits must raise ValueError."""
        with pytest.raises(ValueError, match="fraction"):
            compile_adaptive_precision(lif_neuron, lp_width=8, lp_frac=0, hp_width=16, hp_frac=8)


# ═══════════════════════════════════════════════════════════════════════
# 7. Multi-state-variable neuron
# ═══════════════════════════════════════════════════════════════════════


class TestMultiStateVariable:
    """Verify adaptive precision with multi-state-variable neurons."""

    def test_izhikevich_both_vars_present(self, izhikevich_neuron):
        """Both v and u must appear in all three modules."""
        v = compile_adaptive_precision(izhikevich_neuron, module_name="sc_izh_adapt")
        assert "v_reg" in v
        assert "u_reg" in v
        assert "lp_v_out" in v
        assert "lp_u_out" in v
        assert "hp_v_out" in v
        assert "hp_u_out" in v

    def test_izhikevich_hp_authoritative_both_vars(self, izhikevich_neuron):
        """HP output assignment must be applied to both state variables."""
        v = compile_adaptive_precision(
            izhikevich_neuron,
            module_name="sc_izh_adapt",
            lp_width=16,
            lp_frac=8,
            hp_width=32,
            hp_frac=16,
        )
        assert "v_out <= hp_v_out;" in v
        assert "u_out <= hp_u_out;" in v

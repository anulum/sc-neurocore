#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — tests for tools/gen_vmin_lif_lut.py
import math
import sys
from pathlib import Path

import pytest

from tools.gen_vmin_lif_lut import (
    LUT_RANGE,
    LUT_SIZE,
    Q88_MAX,
    Q88_MIN,
    Q88_SCALE,
    VminLifConfig,
    decode_q88,
    encode_q88,
    emit_lut_verilog_header,
    gen_softplus_lut,
    lut_lookup,
    main,
    softplus_float,
    vmin_lif_step_float,
    vmin_lif_step_q88,
)


# ----- Q8.8 encoding -----


class TestQ88:
    def test_encode_zero(self) -> None:
        assert encode_q88(0.0) == 0

    def test_encode_one(self) -> None:
        assert encode_q88(1.0) == 256

    def test_encode_negative_one(self) -> None:
        assert encode_q88(-1.0) == -256

    def test_encode_quarter(self) -> None:
        assert encode_q88(0.25) == 64

    def test_encode_max_clamp(self) -> None:
        assert encode_q88(1e9) == Q88_MAX

    def test_encode_min_clamp(self) -> None:
        assert encode_q88(-1e9) == Q88_MIN

    def test_decode_round_trip(self) -> None:
        for v in [0.0, 0.25, 0.5, 1.0, -1.0, 5.5, -5.0]:
            assert decode_q88(encode_q88(v)) == pytest.approx(v, abs=1.0 / 256)

    def test_q88_scale_constant(self) -> None:
        assert Q88_SCALE == 256


# ----- softplus reference -----


class TestSoftplus:
    def test_softplus_zero(self) -> None:
        assert softplus_float(0.0, 1.0) == pytest.approx(math.log(2), abs=1e-9)

    def test_softplus_large_positive_linear(self) -> None:
        # softplus(z) ≈ z for large z (matches PyTorch threshold=20 behaviour)
        assert softplus_float(50.0, 1.0) == pytest.approx(50.0, abs=1e-9)

    def test_softplus_negative(self) -> None:
        # softplus(-5) ≈ log(1 + e^-5) ≈ 0.00671
        assert softplus_float(-5.0, 1.0) == pytest.approx(0.00671535, abs=1e-5)

    def test_softplus_beta_scaling(self) -> None:
        # softplus(z, beta=2) = (1/2) * log(1 + e^(2z))
        # at z=1: (1/2) * log(1 + e^2) ≈ 0.5 * log(8.389) ≈ 1.063
        assert softplus_float(1.0, 2.0) == pytest.approx(1.0634640, abs=1e-5)


# ----- LUT generation -----


class TestLUTGeneration:
    def test_lut_size(self) -> None:
        lut = gen_softplus_lut(beta=1.0, size=LUT_SIZE, z_max=LUT_RANGE)
        assert len(lut) == LUT_SIZE

    def test_lut_monotonic(self) -> None:
        lut = gen_softplus_lut(beta=1.0, size=LUT_SIZE, z_max=LUT_RANGE)
        for i in range(len(lut) - 1):
            assert lut[i] <= lut[i + 1], f"non-monotonic at index {i}"

    def test_lut_first_entry_is_log2(self) -> None:
        lut = gen_softplus_lut(beta=1.0, size=LUT_SIZE, z_max=LUT_RANGE)
        # softplus(0) = log(2) ≈ 0.6931
        assert decode_q88(lut[0]) == pytest.approx(math.log(2), abs=1.0 / 256)

    def test_lut_last_entry_near_linear(self) -> None:
        # At z = 16 - step ≈ 15.75, softplus(z) ≈ z
        lut = gen_softplus_lut(beta=1.0, size=LUT_SIZE, z_max=LUT_RANGE)
        last_z = (LUT_SIZE - 1) * (LUT_RANGE / LUT_SIZE)
        assert decode_q88(lut[-1]) == pytest.approx(last_z, abs=0.01)


# ----- LUT lookup with linear interpolation -----


class TestLUTLookup:
    def test_lookup_zero(self) -> None:
        lut = gen_softplus_lut(beta=1.0, size=LUT_SIZE, z_max=LUT_RANGE)
        assert lut_lookup(lut, 0) == lut[0]

    def test_lookup_negative_returns_first(self) -> None:
        lut = gen_softplus_lut(beta=1.0, size=LUT_SIZE, z_max=LUT_RANGE)
        assert lut_lookup(lut, -100) == lut[0]

    def test_lookup_above_range_returns_input(self) -> None:
        # For z >> LUT_RANGE, softplus(z) ≈ z
        lut = gen_softplus_lut(beta=1.0, size=LUT_SIZE, z_max=LUT_RANGE)
        z_q88 = encode_q88(20.0)
        result = lut_lookup(lut, z_q88)
        assert result == z_q88

    def test_lookup_at_lut_endpoints(self) -> None:
        lut = gen_softplus_lut(beta=1.0, size=LUT_SIZE, z_max=LUT_RANGE)
        # z = 0 → first entry
        assert lut_lookup(lut, 0) == lut[0]
        # z = LUT_RANGE → linear extension
        z_q88 = encode_q88(LUT_RANGE)
        assert lut_lookup(lut, z_q88) == z_q88

    def test_lookup_near_upper_bin_returns_last_lut_entry(self) -> None:
        lut = [10, 20, 30, 40]

        assert lut_lookup(lut, encode_q88(15.0), size=len(lut), z_max=16.0) == 40

    def test_lookup_accuracy_vs_float(self) -> None:
        # 1% relative or 0.05 absolute (whichever larger) on the LUT range
        lut = gen_softplus_lut(beta=1.0, size=LUT_SIZE, z_max=LUT_RANGE)
        for z in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]:
            z_q88 = encode_q88(z)
            sp_lut = decode_q88(lut_lookup(lut, z_q88))
            sp_ref = softplus_float(z, 1.0)
            err = abs(sp_lut - sp_ref)
            assert err < max(0.05, 0.01 * sp_ref), (
                f"z={z}: lut={sp_lut:.4f}, ref={sp_ref:.4f}, err={err:.4f}"
            )


# ----- Single-step Vmin_LIF -----


class TestVminLifSingleStep:
    def test_zero_input_zero_state(self) -> None:
        cfg = VminLifConfig()
        lut = gen_softplus_lut(cfg.beta_v_inf, LUT_SIZE, LUT_RANGE)
        # v=0, x=0 → after decay: 0, then softplus floor with v_inf=-5
        # z = 0 - (-5) = 5 → softplus(5) ≈ 5.0067 → v_new = -5 + 5.0067 ≈ 0.0067
        v_next, spike = vmin_lif_step_q88(0, 0, lut, cfg)
        assert spike == 0
        assert decode_q88(v_next) == pytest.approx(0.0067, abs=0.01)

    def test_threshold_crossing_triggers_spike(self) -> None:
        cfg = VminLifConfig()
        lut = gen_softplus_lut(cfg.beta_v_inf, LUT_SIZE, LUT_RANGE)
        # v=0.9, x=0.5 → decay: 0.9*0.75=0.675, +0.5=1.175 > threshold(1.0)
        # JIT eval order: charge→threshold→reset→softplus
        # After reset: v=0. After softplus(0-(-5))=softplus(5)≈5.0067 → v=-5+5.0067≈0.0067
        v_q88 = encode_q88(0.9)
        x_q88 = encode_q88(0.5)
        v_next, spike = vmin_lif_step_q88(v_q88, x_q88, lut, cfg)
        assert spike == 1
        # v_next is v_reset(=0) passed through softplus floor → ~0.0067 in float
        # In Q8.8 that's encode_q88(0.0067) ≈ 1-3 depending on LUT
        assert 0 <= v_next <= 5  # bounded near 0 by softplus

    def test_subthreshold_no_spike(self) -> None:
        cfg = VminLifConfig()
        lut = gen_softplus_lut(cfg.beta_v_inf, LUT_SIZE, LUT_RANGE)
        v_next, spike = vmin_lif_step_q88(encode_q88(0.5), encode_q88(0.1), lut, cfg)
        assert spike == 0

    def test_softplus_floor_prevents_unbounded_negative(self) -> None:
        cfg = VminLifConfig()
        lut = gen_softplus_lut(cfg.beta_v_inf, LUT_SIZE, LUT_RANGE)
        # Even with strongly negative state, softplus floor should bound v
        v_q88 = encode_q88(-4.0)
        v_next, spike = vmin_lif_step_q88(v_q88, encode_q88(-1.0), lut, cfg)
        assert decode_q88(v_next) >= cfg.v_inf  # bounded below by v_inf
        assert spike == 0

    def test_charged_state_saturates_to_q88_max_before_spike(self) -> None:
        cfg = VminLifConfig(v_threshold=1e9)
        lut = gen_softplus_lut(cfg.beta_v_inf, LUT_SIZE, LUT_RANGE)
        v_next, spike = vmin_lif_step_q88(Q88_MAX, Q88_MAX, lut, cfg)

        assert spike == 1
        assert 0 <= v_next <= 5

    def test_charged_state_saturates_to_q88_min(self) -> None:
        cfg = VminLifConfig(v_inf=0.0)
        v_next, spike = vmin_lif_step_q88(Q88_MIN, Q88_MIN, [0], cfg)

        assert spike == 0
        assert v_next == 0

    def test_floor_saturates_to_q88_min(self) -> None:
        cfg = VminLifConfig(v_inf=-1e9)
        v_next, spike = vmin_lif_step_q88(Q88_MIN, Q88_MIN, [-1000], cfg)

        assert spike == 0
        assert v_next == Q88_MIN

    def test_floor_saturates_to_q88_max(self) -> None:
        cfg = VminLifConfig(v_inf=1e9, v_threshold=1e9)
        v_next, spike = vmin_lif_step_q88(0, 0, [1000], cfg)

        assert spike == 0
        assert v_next == Q88_MAX


# ----- Float reference matches PyTorch dynamics -----


class TestFloatReference:
    def test_float_step_matches_pytorch_dynamics(self) -> None:
        # Manual computation of one Vmin_LIF step:
        # v = 0.5, x = 0.3
        # v = 0.5 * 0.75 + 0.3 = 0.675
        # z = 0.675 - (-5) = 5.675
        # softplus(5.675) = log(1 + e^5.675) ≈ 5.6785
        # v = -5 + 5.6785 ≈ 0.6785
        # 0.6785 < 1 → no spike
        cfg = VminLifConfig()
        v_next, spike = vmin_lif_step_float(0.5, 0.3, cfg)
        assert spike == 0
        assert v_next == pytest.approx(0.6785, abs=0.01)

    def test_float_step_threshold_crossing(self) -> None:
        cfg = VminLifConfig()
        v_next, spike = vmin_lif_step_float(0.9, 0.5, cfg)
        # 0.9 * 0.75 + 0.5 = 1.175 ≥ threshold(1.0) → spike, reset to v_reset=0
        # Then softplus floor: v = v_inf + softplus(0 - (-5)) = -5 + softplus(5) ≈ 0.0067
        assert spike == 1
        assert v_next == pytest.approx(0.0067, abs=0.005)


# ----- Trajectory consistency: Q8.8 vs float -----


class TestTrajectoryConsistency:
    def test_constant_input_trajectory(self) -> None:
        cfg = VminLifConfig()
        lut = gen_softplus_lut(cfg.beta_v_inf, LUT_SIZE, LUT_RANGE)

        v_q = encode_q88(-3.0)
        v_f = -3.0
        spikes_q = []
        spikes_f = []
        for _ in range(50):
            v_q, sq = vmin_lif_step_q88(v_q, encode_q88(0.3), lut, cfg)
            v_f, sf = vmin_lif_step_float(v_f, 0.3, cfg)
            spikes_q.append(sq)
            spikes_f.append(sf)

        # Spike count should match within ±1 (due to LUT quantisation near threshold)
        assert abs(sum(spikes_q) - sum(spikes_f)) <= 1

    def test_zero_input_no_runaway(self) -> None:
        cfg = VminLifConfig()
        lut = gen_softplus_lut(cfg.beta_v_inf, LUT_SIZE, LUT_RANGE)
        v_q = 0
        for _ in range(100):
            v_q, _ = vmin_lif_step_q88(v_q, 0, lut, cfg)
        # With zero input, v should converge to a fixed point near 0 (not diverge)
        assert abs(decode_q88(v_q)) < 1.0


# ----- Verilog header emission -----


class TestVerilogHeader:
    def test_header_contains_size_define(self) -> None:
        lut = gen_softplus_lut(1.0, LUT_SIZE, LUT_RANGE)
        header = emit_lut_verilog_header(lut)
        assert f"`define VMIN_LUT_SIZE {LUT_SIZE}" in header

    def test_header_contains_all_entries(self) -> None:
        lut = gen_softplus_lut(1.0, LUT_SIZE, LUT_RANGE)
        header = emit_lut_verilog_header(lut)
        for i in range(LUT_SIZE):
            assert f"`define VMIN_LUT_{i:02d}" in header

    def test_header_uses_signed_q88_literals(self) -> None:
        lut = gen_softplus_lut(1.0, LUT_SIZE, LUT_RANGE)
        header = emit_lut_verilog_header(lut)
        assert "16'sd" in header

    def test_header_has_provenance_comment(self) -> None:
        lut = gen_softplus_lut(1.0, LUT_SIZE, LUT_RANGE)
        header = emit_lut_verilog_header(lut)
        assert "// SPDX-License-Identifier: AGPL-3.0-or-later\n" in header
        assert "// Commercial license available\n" in header
        assert "SPDX-License-Identifier: AGPL-3.0-or-later |" not in header
        assert "Auto-generated" in header
        assert "DO NOT EDIT" in header


class TestCli:
    def test_print_lut_outputs_table(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--print-lut"]) == 0

        output = capsys.readouterr().out
        assert "# Vmin_LIF softplus LUT" in output
        assert "q88=" in output

    def test_out_vh_writes_split_header(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out_path = tmp_path / "vmin_lif_lut.vh"

        assert main(["--out-vh", str(out_path)]) == 0

        output = capsys.readouterr().out
        header = out_path.read_text(encoding="utf-8")
        assert "Written 64 LUT entries" in output
        assert "// SPDX-License-Identifier: AGPL-3.0-or-later\n" in header
        assert "// Commercial license available\n" in header

    def test_demo_outputs_trajectory(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--demo"]) == 0

        output = capsys.readouterr().out
        assert "Demo: 20 steps" in output
        assert "v_float" in output


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

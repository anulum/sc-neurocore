# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMultiPrecision from former test_cosim_emitters.py

"""Focused suite: TestMultiPrecision from former test_cosim_emitters.py."""

from __future__ import annotations

from tests.cosim_emitters_support import *  # noqa: F403

@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestMultiPrecision:
    """Comprehensive multi-precision co-simulation: 9 modes × 5 models.

    Tests organised by angle:
    1. Compilation smoke test — all modes × all models compile and simulate
    2. Accuracy — mV-range modes match Python within 1% for linear models
    3. Zero-current silence — all mV-range modes produce no spikes at I=0
    4. Cross-precision consistency — wider formats produce same or better results
    5. DSP-native modes — Q9.9, Q12.12, Q18.18 exploit hardware widths
    """

    # ── Angle 1: Compilation Smoke Test ──────────────────────────────
    @pytest.mark.parametrize("mode_name", list(_MV_RANGE_MODES.keys()))
    def test_compilation_all_mv_modes(self, mode_name: str) -> None:
        """Every mV-range mode must compile and simulate without errors."""
        dw, frac = _MV_RANGE_MODES[mode_name]
        spikes = _verilog_spike_count_generic("lif", 50, 50.0, dw, frac)
        assert spikes >= 0, f"{mode_name} produced invalid spike count"

    # ── Angle 2: Accuracy (<1%) for Linear Models ────────────────────
    @pytest.mark.parametrize("mode_name", list(_MV_RANGE_MODES.keys()))
    @pytest.mark.parametrize("model_name", _MV_ACCURACY_MODELS)
    def test_accuracy_mv_modes(self, mode_name: str, model_name: str) -> None:
        """mV-range modes must match Python within 1% for linear models."""
        dw, frac = _MV_RANGE_MODES[mode_name]
        py = _python_spike_count(model_name, _N_STEPS, _INPUT_CURRENT)
        vl = _verilog_spike_count_generic(model_name, _N_STEPS, _INPUT_CURRENT, dw, frac)

        gap_pct = abs(py - vl) / max(py, 1) * 100
        print(f"\n  {mode_name} {model_name}: Py={py}, Vl={vl}, gap={gap_pct:.1f}%")

        assert gap_pct < 1.0, f"{mode_name} {model_name}: gap={gap_pct:.1f}% (Py={py}, Vl={vl})"

    # ── Angle 3: Zero-current Silence ────────────────────────────────
    @pytest.mark.parametrize("mode_name", list(_MV_RANGE_MODES.keys()))
    def test_zero_current_silence_mv_modes(self, mode_name: str) -> None:
        """LIF at zero current must produce zero spikes in all mV-range modes."""
        dw, frac = _MV_RANGE_MODES[mode_name]
        spikes = _verilog_spike_count_generic("lif", 50, 0.0, dw, frac)
        assert spikes == 0, f"{mode_name} LIF spiked with zero current: {spikes}"

    # ── Angle 4: Cross-Precision Consistency ─────────────────────────
    def test_cross_precision_lif_monotonicity(self) -> None:
        """Wider precisions must produce the same (or closer to Python) LIF count.

        All mV-range modes should converge to the same spike count as Python
        when the model is linear and the parameters fit.
        """
        py = _python_spike_count("lif", _N_STEPS, _INPUT_CURRENT)
        results: dict[str, int] = {}

        for mode_name, (dw, frac) in sorted(_MV_RANGE_MODES.items(), key=lambda x: x[1][0]):
            vl = _verilog_spike_count_generic("lif", _N_STEPS, _INPUT_CURRENT, dw, frac)
            results[mode_name] = vl

        print(f"\n  Cross-precision LIF (Python={py}):")
        for name, vl in results.items():
            gap = abs(py - vl)
            print(f"    {name:8s}: {vl} spikes (gap={gap})")

        # All should match Python exactly for LIF
        for name, vl in results.items():
            assert vl == py, f"{name} diverges: {vl} vs Python={py}"

    # ── Angle 5: DSP-Native Modes ────────────────────────────────────
    @pytest.mark.parametrize(
        "mode_name,dw,frac",
        [
            ("Q9.9", 18, 9),
            ("Q12.12", 24, 12),
            ("Q14.13", 27, 13),
            ("Q20.12", 32, 12),
            ("Q18.18", 36, 18),
        ],
    )
    def test_dsp_native_lif(self, mode_name: str, dw: int, frac: int) -> None:
        """DSP-native modes must achieve exact Python parity for LIF."""
        py = _python_spike_count("lif", _N_STEPS, _INPUT_CURRENT)
        vl = _verilog_spike_count_generic("lif", _N_STEPS, _INPUT_CURRENT, dw, frac)
        assert vl == py, f"{mode_name} ({dw}-bit) diverges: Verilog={vl}, Python={py}"

    # ── Angle 6: Narrow-Range Modes Compilation ──────────────────────
    @pytest.mark.parametrize(
        "mode_name,dw,frac",
        [
            ("Q1.7", 8, 7),
            ("Q1.15", 16, 15),
            ("Q4.12", 16, 12),
        ],
    )
    def test_narrow_range_compile_smoke(self, mode_name: str, dw: int, frac: int) -> None:
        """Narrow-range modes must compile without errors for any model.

        These modes may not match Python due to range overflow, but the
        compiler and simulator must not crash.
        """
        # Use resonate_fire: its initial state is closer to zero
        spikes = _verilog_spike_count_generic(
            "resonate_fire",
            50,
            10.0,
            dw,
            frac,
        )
        assert spikes >= 0

    # ── Angle 7: Precision Report API ────────────────────────────────
    def test_q88_class_all_modes(self) -> None:
        """Q88 dataclass works correctly at all 9 precision configurations."""
        from sc_neurocore.compiler.equation_compiler import Q88

        for mode_name, (dw, frac) in _ALL_MODES.items():
            q = Q88(data_width=dw, fraction=frac)
            assert q.integer_bits == dw - frac - 1
            assert q.resolution == pytest.approx(1.0 / (1 << frac))

            # Encode/decode roundtrip
            for test_val in [0.0, 1.0, -1.0, 0.5]:
                if q.min_value <= test_val <= q.max_value:
                    encoded = q.encode(test_val)
                    assert isinstance(encoded, int)
                    literal = q.encode_signed_literal(test_val)
                    assert literal.endswith(f"'sd{encoded}")

            # Precision report must not crash
            report = q.precision_report(dt=0.01, params={"test": 1.0})
            assert "Fixed-point format" in report

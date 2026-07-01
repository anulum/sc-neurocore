# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for the rebuilt HLS C++ exporter

"""Tests for the ap_fixed HLS C++ exporter, including a g++ compile check."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from sc_neurocore.compiler.intelligence.hls_export import generate_hls_cpp

# Stub headers so a host g++ can syntax-check the generated Vitis HLS unit
# (ap_fixed collapses to double; hls_math forwards to <cmath>).
_AP_FIXED_STUB = """#pragma once
#include <cmath>
template <int W, int I> using ap_fixed = double;
"""

_HLS_MATH_STUB = """#pragma once
#include <cmath>
namespace hls {
inline double exp(double x) { return std::exp(x); }
inline double log(double x) { return std::log(x); }
inline double sqrt(double x) { return std::sqrt(x); }
inline double cbrt(double x) { return std::cbrt(x); }
inline double tanh(double x) { return std::tanh(x); }
inline double cosh(double x) { return std::cosh(x); }
inline double sin(double x) { return std::sin(x); }
inline double cos(double x) { return std::cos(x); }
inline double abs(double x) { return std::fabs(x); }
}
"""


class TestEulerIntegration:
    def test_derivative_is_euler_integrated(self) -> None:
        cpp = generate_hls_cpp("sc_lif", {"v": "-v / tau + I"})
        assert "fp_t d_v = " in cpp
        assert "fp_t v_next = v + dt * d_v;" in cpp

    def test_input_current_lowered_to_I_t(self) -> None:
        cpp = generate_hls_cpp("sc_lif", {"v": "I"})
        assert "fp_t d_v = I_t;" in cpp

    def test_dt_is_a_parameter(self) -> None:
        cpp = generate_hls_cpp("sc_lif", {"v": "I"})
        assert "fp_t dt," in cpp


class TestFreeVariables:
    def test_free_vars_become_inputs(self) -> None:
        cpp = generate_hls_cpp("sc_lif", {"v": "-v / tau + I - leak * v"})
        assert "fp_t tau," in cpp
        assert "fp_t leak," in cpp

    def test_state_vars_not_declared_as_inputs(self) -> None:
        cpp = generate_hls_cpp("sc_izh", {"v": "u + I", "u": "v"})
        # v and u are &-referenced state, never plain input params.
        assert "fp_t &v," in cpp
        assert "fp_t &u," in cpp


class TestThresholdAndReset:
    def test_threshold_is_configurable(self) -> None:
        cpp = generate_hls_cpp("sc_lif", {"v": "I"}, threshold=2.5)
        assert "const fp_t V_THRESH = fp_t(2.5);" in cpp

    def test_membrane_resets_by_subtracting_threshold(self) -> None:
        cpp = generate_hls_cpp("sc_lif", {"v": "I"})
        assert "v = spike_out ? (v_next - V_THRESH) : v_next;" in cpp

    def test_spike_detection_on_first_state_var(self) -> None:
        cpp = generate_hls_cpp("sc_izh", {"v": "u + I", "u": "v"})
        assert "spike_out = (v_next > V_THRESH);" in cpp
        # The non-membrane variable updates without a reset.
        assert "u = u_next;" in cpp


class TestHelpers:
    def test_sigmoid_helper_emitted_when_used(self) -> None:
        cpp = generate_hls_cpp("sc_x", {"v": "sigmoid(v)"})
        assert "static inline fp_t sc_sigmoid(fp_t x)" in cpp

    def test_exprel_helper_emitted_when_used(self) -> None:
        cpp = generate_hls_cpp("sc_x", {"v": "exprel(v)"})
        assert "static inline fp_t sc_exprel(fp_t x)" in cpp

    def test_helpers_absent_when_unused(self) -> None:
        cpp = generate_hls_cpp("sc_lif", {"v": "I"})
        assert "sc_sigmoid" not in cpp
        assert "sc_exprel" not in cpp


class TestStructureAndBackends:
    def test_include_guard_and_typedef(self) -> None:
        cpp = generate_hls_cpp("sc_lif", {"v": "I"}, data_width=16, fraction=8)
        assert "#ifndef SC_LIF_HLS_H" in cpp
        assert "#endif // SC_LIF_HLS_H" in cpp
        assert "typedef ap_fixed<16,8> fp_t;" in cpp

    def test_state_struct(self) -> None:
        cpp = generate_hls_cpp("sc_izh", {"v": "u", "u": "v"})
        assert "struct sc_izh_state {" in cpp
        assert "fp_t v;" in cpp
        assert "fp_t u;" in cpp

    def test_vitis_pipeline_pragma(self) -> None:
        cpp = generate_hls_cpp("sc_lif", {"v": "I"}, hls_tool="vitis")
        assert "#pragma HLS PIPELINE II=1" in cpp

    def test_catapult_backend(self) -> None:
        cpp = generate_hls_cpp("sc_lif", {"v": "I"}, hls_tool="catapult")
        assert "Catapult" in cpp
        assert "#pragma HLS PIPELINE" not in cpp


@pytest.mark.skipif(shutil.which("g++") is None, reason="g++ not available")
class TestCompiles:
    """The generated HLS C++ must actually compile (with ap_fixed/hls_math stubs)."""

    def _compile(self, tmp_path: Path, cpp: str, name: str) -> subprocess.CompletedProcess[str]:
        (tmp_path / "ap_fixed.h").write_text(_AP_FIXED_STUB)
        (tmp_path / "hls_math.h").write_text(_HLS_MATH_STUB)
        (tmp_path / f"{name}.h").write_text(cpp)
        main = tmp_path / "main.cpp"
        main.write_text(f'#include "{name}.h"\nint main() {{ return 0; }}\n')
        return subprocess.run(
            [
                "g++",
                "-fsyntax-only",
                "-std=c++14",
                "-Wno-unknown-pragmas",
                "-I",
                str(tmp_path),
                str(main),
            ],
            capture_output=True,
            text=True,
        )

    def test_lif_compiles(self, tmp_path: Path) -> None:
        cpp = generate_hls_cpp("sc_lif", {"v": "-v / tau + I - leak * v"})
        result = self._compile(tmp_path, cpp, "sc_lif")
        assert result.returncode == 0, result.stderr

    def test_izhikevich_compiles(self, tmp_path: Path) -> None:
        cpp = generate_hls_cpp(
            "sc_izh",
            {"v": "0.04 * v * v + 5.0 * v + 140.0 - u + I", "u": "a * (b * v - u)"},
        )
        result = self._compile(tmp_path, cpp, "sc_izh")
        assert result.returncode == 0, result.stderr

    def test_transcendental_compiles(self, tmp_path: Path) -> None:
        cpp = generate_hls_cpp("sc_x", {"v": "tanh(v) + sigmoid(v) + exprel(I) + exp(v)"})
        result = self._compile(tmp_path, cpp, "sc_x")
        assert result.returncode == 0, result.stderr

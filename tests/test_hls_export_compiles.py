# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCompiles from former test_hls_export.py

"""Focused suite: TestCompiles from former test_hls_export.py."""

from __future__ import annotations

from tests.hls_export_support import *  # noqa: F403

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

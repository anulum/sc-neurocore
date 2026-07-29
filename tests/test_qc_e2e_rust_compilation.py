# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRustCompilation from former test_qc_e2e.py

"""Focused suite: TestRustCompilation from former test_qc_e2e.py."""

from __future__ import annotations

from tests.qc_e2e_support import *  # noqa: F403


class TestRustCompilation:
    """Verify all Rust files compile and pass tests."""

    @pytest.mark.parametrize("rs_file", ["spin_pool.rs", "radical_pair.rs", "kane_mapper.rs"])
    def test_rust_compiles_and_tests_pass(self, rs_file: str) -> None:
        rs_path = _QC_DIR / rs_file
        assert rs_path.is_file(), f"committed Rust quantum-cognition source missing: {rs_path}"

        bin_name = rs_file.replace(".rs", "_test")
        out_path = f"/tmp/{bin_name}"

        result = subprocess.run(
            ["rustc", "--test", str(rs_path), "-o", out_path, "-C", "opt-level=2"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"Compilation failed:\n{result.stderr}"

        result = subprocess.run(
            [out_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"Tests failed:\n{result.stdout}\n{result.stderr}"
        assert "test result: ok" in result.stdout

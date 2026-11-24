# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBitExactParity from former test_wilson_cowan_parity.py

"""Focused suite: TestBitExactParity from former test_wilson_cowan_parity.py."""

from __future__ import annotations

import numpy as np
from tests.wilson_cowan_parity_support import (
    _run_python,
    _run_rust,
)


class TestBitExactParity:
    """Python and Rust Wilson-Cowan traces agree to the last ulp.

    Wilson-Cowan is noise-free; Python and Rust traces must match within
    a small last-ulp envelope across compiler and Python-version math paths.
    """

    def test_parity_zero_input(self) -> None:
        """Rust traces under zero drive match the Python primary within 2e-14 for the E and I traces."""
        n = 5_000
        e_py, i_py = _run_python(np.zeros(n))
        e_rs, i_rs = _run_rust(np.zeros(n))
        assert np.allclose(e_py, e_rs, atol=2e-14, rtol=0)
        assert np.allclose(i_py, i_rs, atol=2e-14, rtol=0)

    def test_parity_constant_drive(self) -> None:
        """Rust traces under constant drive match the Python primary within 2e-14 for the E and I traces."""
        n = 5_000
        ext = np.full(n, 1.5)
        e_py, i_py = _run_python(ext)
        e_rs, i_rs = _run_rust(ext)
        assert np.allclose(e_py, e_rs, atol=2e-14, rtol=0)
        assert np.allclose(i_py, i_rs, atol=2e-14, rtol=0)

    def test_parity_time_varying_drive(self) -> None:
        """Rust traces under a sinusoidal drive match the Python primary within 2e-14 for the E and I traces."""
        n = 3_000
        ext = np.sin(np.linspace(0, 10 * np.pi, n)) * 2.0
        e_py, i_py = _run_python(ext)
        e_rs, i_rs = _run_rust(ext)
        assert np.allclose(e_py, e_rs, atol=2e-14, rtol=0)
        assert np.allclose(i_py, i_rs, atol=2e-14, rtol=0)

    def test_parity_step_function_drive(self) -> None:
        """Sharp transitions are the hardest test for integration parity."""
        n = 4_000
        ext = np.zeros(n)
        ext[1_000:2_000] = 5.0
        ext[3_000:] = -2.0
        e_py, i_py = _run_python(ext)
        e_rs, i_rs = _run_rust(ext)
        assert np.allclose(e_py, e_rs, atol=2e-14, rtol=0)
        assert np.allclose(i_py, i_rs, atol=2e-14, rtol=0)

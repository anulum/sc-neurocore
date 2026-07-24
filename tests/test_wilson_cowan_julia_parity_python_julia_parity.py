# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPythonJuliaParity from former test_wilson_cowan_julia_parity.py

"""Focused suite: TestPythonJuliaParity from former test_wilson_cowan_julia_parity.py."""

from __future__ import annotations

from tests.wilson_cowan_julia_parity_support import *  # noqa: F403


class TestPythonJuliaParity:
    """Julia trajectory must match Python primary at machine-epsilon."""

    def test_parity_zero_input(self):
        n = 3_000
        e_py, i_py = _run_python(np.zeros(n))
        e_jl, i_jl = _run_julia(np.zeros(n))
        assert np.allclose(e_py, e_jl, atol=1e-14, rtol=0)
        assert np.allclose(i_py, i_jl, atol=1e-14, rtol=0)

    def test_parity_constant_drive(self):
        n = 3_000
        ext = np.full(n, 1.5)
        e_py, i_py = _run_python(ext)
        e_jl, i_jl = _run_julia(ext)
        assert np.allclose(e_py, e_jl, atol=1e-14, rtol=0)
        assert np.allclose(i_py, i_jl, atol=1e-14, rtol=0)

    def test_parity_sinusoid_drive(self):
        n = 2_000
        ext = np.sin(np.linspace(0, 8 * np.pi, n)) * 2.0
        e_py, i_py = _run_python(ext)
        e_jl, i_jl = _run_julia(ext)
        assert np.allclose(e_py, e_jl, atol=1e-14, rtol=0)
        assert np.allclose(i_py, i_jl, atol=1e-14, rtol=0)

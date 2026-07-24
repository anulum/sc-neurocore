# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPythonMojoParity from former test_wilson_cowan_mojo_parity.py

"""Focused suite: TestPythonMojoParity from former test_wilson_cowan_mojo_parity.py."""

from __future__ import annotations

from tests.wilson_cowan_mojo_parity_support import *  # noqa: F403


class TestPythonMojoParity:
    def test_parity_zero_input(self):
        n = 3_000
        e_py, i_py = _run_python(np.zeros(n))
        e_mj, i_mj = _run_mojo(np.zeros(n))
        assert np.allclose(e_py, e_mj, atol=1e-9, rtol=0)
        assert np.allclose(i_py, i_mj, atol=1e-9, rtol=0)

    def test_parity_constant_drive(self):
        n = 3_000
        ext = np.full(n, 2.0)
        e_py, i_py = _run_python(ext)
        e_mj, i_mj = _run_mojo(ext)
        # Nonlinear sigmoid amplifies the libm-vs-std-exp ulp drift
        # quickly; measured drift is bounded by ~1e-12 on short windows
        # and ~1e-9 over thousands of steps.
        assert np.allclose(e_py[:10], e_mj[:10], atol=1e-12, rtol=0)
        assert np.allclose(e_py, e_mj, atol=1e-9, rtol=0)
        assert np.allclose(i_py, i_mj, atol=1e-9, rtol=0)

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPythonGoParity from former test_wilson_cowan_go_parity.py

"""Focused suite: TestPythonGoParity from former test_wilson_cowan_go_parity.py."""

from __future__ import annotations

from tests.wilson_cowan_go_parity_support import *  # noqa: F403


class TestPythonGoParity:
    def test_parity_zero_input(self):
        n = 3_000
        e_py, i_py = _run_python(np.zeros(n))
        e_go, i_go = _run_go(np.zeros(n))
        assert np.allclose(e_py, e_go, atol=1e-14, rtol=0)
        assert np.allclose(i_py, i_go, atol=1e-14, rtol=0)

    def test_parity_constant_drive(self):
        n = 3_000
        ext = np.full(n, 2.0)
        e_py, i_py = _run_python(ext)
        e_go, i_go = _run_go(ext)
        assert np.allclose(e_py, e_go, atol=1e-14, rtol=0)
        assert np.allclose(i_py, i_go, atol=1e-14, rtol=0)

    def test_parity_ramp_drive(self):
        n = 3_000
        ext = np.linspace(-1.0, 4.0, n)
        e_py, i_py = _run_python(ext)
        e_go, i_go = _run_go(ext)
        assert np.allclose(e_py, e_go, atol=1e-14, rtol=0)
        assert np.allclose(i_py, i_go, atol=1e-14, rtol=0)

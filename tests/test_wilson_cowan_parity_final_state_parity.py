# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFinalStateParity from former test_wilson_cowan_parity.py

"""Focused suite: TestFinalStateParity from former test_wilson_cowan_parity.py."""

from __future__ import annotations

from tests.wilson_cowan_parity_support import *  # noqa: F403


class TestFinalStateParity:
    def test_rust_e_final_matches_trace_last(self):
        n = 2_000
        ext = np.full(n, 1.0)
        out = py_wilson_cowan_simulate(
            0.1,
            0.05,
            DEFAULT_PARAMS["w_ee"],
            DEFAULT_PARAMS["w_ei"],
            DEFAULT_PARAMS["w_ie"],
            DEFAULT_PARAMS["w_ii"],
            DEFAULT_PARAMS["tau_e"],
            DEFAULT_PARAMS["tau_i"],
            DEFAULT_PARAMS["a"],
            DEFAULT_PARAMS["theta"],
            DEFAULT_PARAMS["dt"],
            ext,
        )
        assert out["e_final"] == out["e"][-1]
        assert out["i_final"] == out["i"][-1]

    def test_python_rust_final_state_match(self):
        n = 3_000
        ext = np.full(n, 2.5)
        e_py, i_py = _run_python(ext)
        e_rs, i_rs = _run_rust(ext)
        assert abs(e_py[-1] - e_rs[-1]) < 2e-14
        assert abs(i_py[-1] - i_rs[-1]) < 2e-14

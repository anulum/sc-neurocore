# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRustJuliaCrossParity from former test_wilson_cowan_julia_parity.py

"""Focused suite: TestRustJuliaCrossParity from former test_wilson_cowan_julia_parity.py."""

from __future__ import annotations

from tests.wilson_cowan_julia_parity_support import *  # noqa: F403


class TestRustJuliaCrossParity:
    """Rust + Julia must agree to machine epsilon under identical inputs."""

    def test_rust_julia_identical(self):
        rs = pytest.importorskip(
            "sc_neurocore_engine", reason="Rust engine required"
        ).py_wilson_cowan_simulate
        n = 3_000
        ext = np.linspace(-1.0, 4.0, n)
        r_out = rs(
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
        j_out = simulate_wilson_cowan(
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
        assert np.allclose(r_out["e"], j_out["e"], atol=1e-14, rtol=0)
        assert np.allclose(r_out["i"], j_out["i"], atol=1e-14, rtol=0)
        assert abs(r_out["e_final"] - j_out["e_final"]) < 1e-14
        assert abs(r_out["i_final"] - j_out["i_final"]) < 1e-14

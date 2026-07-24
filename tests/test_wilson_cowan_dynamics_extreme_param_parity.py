# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestExtremeParamParity from former test_wilson_cowan_dynamics.py

"""Focused suite: TestExtremeParamParity from former test_wilson_cowan_dynamics.py."""

from __future__ import annotations

from tests.wilson_cowan_dynamics_support import *  # noqa: F403


class TestExtremeParamParity:
    """Rust simulator must track Python primary bit-exact across
    extreme parameter regimes."""

    rust = pytest.importorskip(
        "sc_neurocore_engine", reason="Rust engine required"
    ).py_wilson_cowan_simulate

    @pytest.mark.parametrize(
        "params",
        [
            dict(tau_e=0.1, tau_i=0.1, dt=0.01),  # fast dynamics
            dict(tau_e=10.0, tau_i=20.0, dt=0.5),  # slow dynamics, coarse dt
            dict(a=0.5, theta=0.0),  # shallow sigmoid
            dict(a=3.0, theta=8.0),  # steep sigmoid
            dict(w_ee=20.0, w_ei=1.0),  # strong excitation
        ],
    )
    def test_parity_extreme_params(self, params):
        p = {**DEFAULT_PARAMS, **params}
        n = 2_000
        ext = np.linspace(-2.0, 5.0, n)
        u = WilsonCowanUnit(**p)
        e_py = np.empty(n)
        for t in range(n):
            u.step(float(ext[t]))
            e_py[t] = u.e
        out = self.rust(
            0.1,
            0.05,
            p["w_ee"],
            p["w_ei"],
            p["w_ie"],
            p["w_ii"],
            p["tau_e"],
            p["tau_i"],
            p["a"],
            p["theta"],
            p["dt"],
            ext,
        )
        assert np.allclose(e_py, out["e"], atol=1e-14, rtol=0), f"drift under params={params}"

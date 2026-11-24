# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRustMojoCrossParity from former test_wilson_cowan_mojo_parity.py

"""Focused suite: TestRustMojoCrossParity from former test_wilson_cowan_mojo_parity.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.accel.mojo.wilson_cowan import (
    simulate_wilson_cowan,
)
from tests.wilson_cowan_mojo_parity_support import (
    DEFAULT_PARAMS,
)


class TestRustMojoCrossParity:
    """The Mojo and Rust Wilson-Cowan simulators against each other."""

    def test_rust_mojo_within_ulp_drift(self) -> None:
        """Rust and Mojo traces under a sinusoidal drive agree within 1e-9."""
        rs = pytest.importorskip(
            "sc_neurocore_engine", reason="Rust engine required"
        ).py_wilson_cowan_simulate
        n = 3_000
        ext = np.sin(np.linspace(0, 6 * np.pi, n)) * 3.0
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
        m_out = simulate_wilson_cowan(
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
        assert np.allclose(r_out["e"], m_out["e"], atol=1e-9, rtol=0)
        assert np.allclose(r_out["i"], m_out["i"], atol=1e-9, rtol=0)

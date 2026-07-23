# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLongRunStability from former test_wilson_cowan_dynamics.py

"""Focused suite: TestLongRunStability from former test_wilson_cowan_dynamics.py."""

from __future__ import annotations

from tests.wilson_cowan_dynamics_support import *  # noqa: F403

class TestLongRunStability:
    """No NaN, no inf, no drift outside the published envelope over very
    long simulations. `feedback_module_standard_attnres` requires
    algorithm/parity/**stability** tests; this section provides the
    third leg. We run the Rust simulator because the Python primary
    would be too slow for 1 M-step sweeps in the test suite."""

    rust = pytest.importorskip(
        "sc_neurocore_engine", reason="Rust engine required"
    ).py_wilson_cowan_simulate

    @pytest.mark.parametrize(
        "drive,dt",
        [
            (0.0, 0.1),
            (3.0, 0.1),
            (1.25, 0.05),  # oscillator regime, smaller dt
            (10.0, 0.2),  # strong drive, coarser dt
        ],
    )
    def test_no_nan_no_inf_over_1M_steps(self, drive, dt):
        """Long-run integration must stay finite. 1 M steps is overkill
        for the published parameter range but catches accumulated
        f64 round-off that would show up only at large N."""
        n = 1_000_000
        ext = np.full(n, drive, dtype=np.float64)
        out = self.rust(
            0.1,
            0.05,
            10.0,
            6.0,
            10.0,
            1.0,
            1.0,
            2.0,
            1.2,
            4.0,
            dt,
            ext,
        )
        e, i = out["e"], out["i"]
        assert np.isfinite(e).all(), f"E went non-finite at drive={drive}, dt={dt}"
        assert np.isfinite(i).all(), f"I went non-finite at drive={drive}, dt={dt}"
        # Published envelope: [-β, 1-β] + Euler relaxation slack.
        baseline = 1.0 / (1.0 + math.exp(1.2 * 4.0))  # a=1.2, θ=4.0
        lo = -baseline - 1e-6
        hi = 1.0 + baseline + 1e-6
        assert e.min() >= lo and e.max() <= hi, (
            f"E out of envelope at drive={drive}, dt={dt}: [{e.min():.4f}, {e.max():.4f}]"
        )
        assert i.min() >= lo and i.max() <= hi

    def test_steady_state_convergence_1M_steps(self):
        """Under constant drive outside the oscillator regime, the state
        must settle. Check that the trailing 100k steps show very small
        variance (steady state reached)."""
        n = 1_000_000
        ext = np.full(n, 5.0, dtype=np.float64)
        out = self.rust(
            0.1,
            0.05,
            10.0,
            6.0,
            10.0,
            1.0,
            1.0,
            2.0,
            1.2,
            4.0,
            0.1,
            ext,
        )
        tail_std = float(np.std(out["e"][-100_000:]))
        assert tail_std < 1e-6, f"E should have settled to a fixed point; tail std = {tail_std:.2e}"

    def test_time_reversibility_against_short_run(self):
        """Simulator must be state-function of caller inputs only — so two
        independent 500k-step runs from identical init + drive must
        produce bit-identical final states (no hidden state, no
        accumulator leakage across calls)."""
        n = 500_000
        ext = np.full(n, 2.0, dtype=np.float64)
        out_a = self.rust(0.1, 0.05, 10.0, 6.0, 10.0, 1.0, 1.0, 2.0, 1.2, 4.0, 0.1, ext)
        out_b = self.rust(0.1, 0.05, 10.0, 6.0, 10.0, 1.0, 1.0, 2.0, 1.2, 4.0, 0.1, ext)
        assert out_a["e_final"] == out_b["e_final"]
        assert out_a["i_final"] == out_b["i_final"]

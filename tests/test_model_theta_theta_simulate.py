# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestThetaSimulate from former test_model_theta.py

"""Focused suite: TestThetaSimulate from former test_model_theta.py."""

from __future__ import annotations

from tests.model_theta_support import *  # noqa: F403

class TestThetaSimulate:
    """Engineering-verification surface for ``ThetaNeuron.simulate``."""

    def test_simulate_python_returns_finite_trace(self) -> None:
        n = ThetaNeuron()
        trace, spikes = n.simulate(1000, current=1.0, backend="python")
        assert trace.shape == (1000,)
        assert np.all(np.isfinite(trace))
        assert spikes >= 1
        assert n.theta == float(trace[-1])

    def test_simulate_rust_matches_python(self) -> None:
        assert backends._HAS_RUST
        py = ThetaNeuron()
        rs = ThetaNeuron()
        tr_py, sp_py = py.simulate(1000, current=1.0, backend="python")
        tr_rs, sp_rs = rs.simulate(1000, current=1.0, backend="rust")
        assert sp_py == sp_rs
        assert np.array_equal(tr_py, tr_rs)

    def test_simulate_rust_rejects_non_default(self) -> None:
        assert backends._HAS_RUST
        n = ThetaNeuron(dt=0.02)
        with pytest.raises(RuntimeError, match="factory-default"):
            n.simulate(10, current=0.0, backend="rust")

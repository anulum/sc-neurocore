# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestConnorStevensNeuronSimulate from former test_model_connor_stevens.py

"""Focused suite: TestConnorStevensNeuronSimulate from former test_model_connor_stevens.py."""

from __future__ import annotations

from tests.model_connor_stevens_support import *  # noqa: F403

class TestConnorStevensNeuronSimulate:
    """Engineering-verification surface for ``ConnorStevensNeuron.simulate``."""

    def test_simulate_python_returns_finite_trace(self) -> None:
        n = ConnorStevensNeuron()
        trace, spikes = n.simulate(1000, current=10.0, backend="python")
        assert trace.shape == (1000,)
        assert np.all(np.isfinite(trace))
        assert spikes >= 1

    def test_simulate_rust_matches_or_ulp_python(self) -> None:
        pytest.importorskip("sc_neurocore_engine", reason="Rust engine not built")
        py = ConnorStevensNeuron()
        rs = ConnorStevensNeuron()
        tr_py, sp_py = py.simulate(1000, current=10.0, backend="python")
        tr_rs, sp_rs = rs.simulate(1000, current=10.0, backend="rust")
        assert sp_py == sp_rs
        max_diff = float(np.max(np.abs(tr_py - tr_rs)))
        assert max_diff < 1e-9

    def test_simulate_rust_rejects_non_default(self) -> None:
        pytest.importorskip("sc_neurocore_engine", reason="Rust engine not built")
        # force non-default via a constructor override that every model accepts
        try:
            n = (
                ConnorStevensNeuron(dt=0.02)
                if "dt" in ConnorStevensNeuron.__dataclass_fields__
                else ConnorStevensNeuron()
            )
            if "dt" not in ConnorStevensNeuron.__dataclass_fields__:
                pytest.skip("no dt field")
        except TypeError:
            pytest.skip("cannot override defaults")
        with pytest.raises(RuntimeError, match="factory-default"):
            n.simulate(10, current=0.0, backend="rust")

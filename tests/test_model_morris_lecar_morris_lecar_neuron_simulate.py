# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMorrisLecarNeuronSimulate from former test_model_morris_lecar.py

"""Focused suite: TestMorrisLecarNeuronSimulate from former test_model_morris_lecar.py."""

from __future__ import annotations

from tests.model_morris_lecar_support import *  # noqa: F403

class TestMorrisLecarNeuronSimulate:
    """Engineering-verification surface for ``MorrisLecarNeuron.simulate``."""

    def test_simulate_python_returns_finite_trace(self) -> None:
        n = MorrisLecarNeuron()
        trace, spikes = n.simulate(1000, current=80.0, backend="python")
        assert trace.shape == (1000,)
        assert np.all(np.isfinite(trace))
        assert spikes >= 1

    def test_simulate_rust_matches_or_ulp_python(self) -> None:
        pytest.importorskip("sc_neurocore_engine", reason="Rust engine not built")
        py = MorrisLecarNeuron()
        rs = MorrisLecarNeuron()
        tr_py, sp_py = py.simulate(1000, current=80.0, backend="python")
        tr_rs, sp_rs = rs.simulate(1000, current=80.0, backend="rust")
        assert sp_py == sp_rs
        max_diff = float(np.max(np.abs(tr_py - tr_rs)))
        assert max_diff < 1e-9

    def test_simulate_rust_rejects_non_default(self) -> None:
        pytest.importorskip("sc_neurocore_engine", reason="Rust engine not built")
        # force non-default via a constructor override that every model accepts
        try:
            n = (
                MorrisLecarNeuron(dt=0.02)
                if "dt" in MorrisLecarNeuron.__dataclass_fields__
                else MorrisLecarNeuron()
            )
            if "dt" not in MorrisLecarNeuron.__dataclass_fields__:
                pytest.skip("no dt field")
        except TypeError:
            pytest.skip("cannot override defaults")
        with pytest.raises(RuntimeError, match="factory-default"):
            n.simulate(10, current=0.0, backend="rust")

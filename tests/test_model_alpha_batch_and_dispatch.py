# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBatchAndDispatch from former test_model_alpha.py

"""Focused suite: TestBatchAndDispatch from former test_model_alpha.py."""

from __future__ import annotations

from tests.model_alpha_support import *  # noqa: F403


class TestBatchAndDispatch:
    """The maintained batch lane matches the scalar golden loop."""

    def test_batch_matches_scalar_step_loop(self) -> None:
        exc = 1.5 + 0.8 * np.sin(np.arange(256) * 0.037)
        inh = 0.6 + 0.3 * np.cos(np.arange(256) * 0.021)
        scalar = AlphaNeuron()
        expected: dict[str, list[float]] = {
            key: [] for key in ("v", "a_exc", "i_exc", "a_inh", "i_inh")
        }
        expected_spikes = 0
        for exc_value, inh_value in zip(exc, inh):
            expected_spikes += scalar.step(float(exc_value), float(inh_value))
            for key in expected:
                expected[key].append(getattr(scalar, key))
        batch = AlphaNeuron().simulate(exc, inh, backend="python")
        for key in expected:
            np.testing.assert_allclose(batch[key], expected[key], rtol=0.0, atol=0.0)
        assert batch["spike_count"] == expected_spikes

    def test_scalar_inhibitory_broadcast_matches_vector(self) -> None:
        exc = np.full(64, 2.0)
        vector = AlphaNeuron().simulate(exc, np.full(64, 0.5), backend="python")
        scalar = AlphaNeuron().simulate(exc, 0.5, backend="python")
        np.testing.assert_array_equal(vector["v"], scalar["v"])

    def test_empty_batch_returns_initial_state(self) -> None:
        result = AlphaNeuron(v=0.1, a_exc=0.2).simulate([], backend="python")
        assert cast(npt.NDArray[np.float64], result["v"]).size == 0
        assert result["v_final"] == 0.1
        assert result["a_exc_final"] == 0.2
        assert result["spike_count"] == 0

    def test_simulate_writes_back_final_state(self) -> None:
        n = AlphaNeuron()
        result = n.simulate(np.full(200, 2.0), backend="python")
        assert n.v == result["v_final"]
        assert n.a_exc == result["a_exc_final"]

    def test_long_varied_run_is_finite_and_deterministic(self) -> None:
        exc = 2.0 + 0.5 * np.sin(np.arange(20_000, dtype=np.float64) * 0.013)
        inh = 0.8 + 0.2 * np.cos(np.arange(20_000, dtype=np.float64) * 0.007)
        first = AlphaNeuron().simulate(exc, inh, backend="python")
        second = AlphaNeuron().simulate(exc, inh, backend="python")
        assert np.isfinite(first["v"]).all()
        np.testing.assert_array_equal(first["v"], second["v"])
        np.testing.assert_array_equal(first["theta" if False else "i_exc"], second["i_exc"])

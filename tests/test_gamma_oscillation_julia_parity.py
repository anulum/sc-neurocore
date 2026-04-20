# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for PING gamma oscillation circuit (Julia Parity)

import numpy as np
import pytest

from sc_neurocore.network.gamma_oscillation import (
    _HAS_JULIA_PING_STEP,
    PINGCircuit,
)


@pytest.mark.skipif(
    not _HAS_JULIA_PING_STEP,
    reason="Julia kernel not loaded",
)
class TestPythonJuliaParity:
    """Per-population spike rates match between the NumPy and Julia
    backends within a tight tolerance.
    """

    @pytest.mark.parametrize(
        "n_e,n_i",
        [(80, 20), (400, 100), (1000, 250)],
    )
    def test_population_rates_match(self, n_e, n_i):
        ping_py = PINGCircuit(
            n_excitatory=n_e,
            n_inhibitory=n_i,
            seed=42,
            backend="python",
        )
        ping_julia = PINGCircuit(
            n_excitatory=n_e,
            n_inhibitory=n_i,
            seed=42,
            backend="julia",
        )
        for _ in range(1000):
            ping_py.step(dt=0.1)
            ping_julia.step(dt=0.1)
        e_py = e_julia = i_py = i_julia = 0
        for _ in range(5000):
            se_py, si_py = ping_py.step(dt=0.1)
            se_julia, si_julia = ping_julia.step(dt=0.1)
            e_py += int(np.count_nonzero(se_py))
            e_julia += int(np.count_nonzero(se_julia))
            i_py += int(np.count_nonzero(si_py))
            i_julia += int(np.count_nonzero(si_julia))
        rate_e_py = e_py / n_e
        rate_e_julia = e_julia / n_e
        rate_i_py = i_py / n_i
        rate_i_julia = i_julia / n_i
        assert abs(rate_e_py - rate_e_julia) / max(rate_e_py, 1e-3) < 0.10, (
            f"E rate mismatch py={rate_e_py:.3f} julia={rate_e_julia:.3f}"
        )
        assert abs(rate_i_py - rate_i_julia) / max(rate_i_py, 1e-3) < 0.10, (
            f"I rate mismatch py={rate_i_py:.3f} julia={rate_i_julia:.3f}"
        )

    def test_dominant_frequency_matches_across_backends(self):
        ping_py = PINGCircuit(
            n_excitatory=400,
            n_inhibitory=100,
            seed=42,
            backend="python",
        )
        ping_julia = PINGCircuit(
            n_excitatory=400,
            n_inhibitory=100,
            seed=42,
            backend="julia",
        )
        for _ in range(2000):
            ping_py.step(dt=0.1)
            ping_julia.step(dt=0.1)
        sp_py, sp_julia = [], []
        for _ in range(8000):
            se_py, _ = ping_py.step(dt=0.1)
            sp_py.append(se_py)
            se_julia, _ = ping_julia.step(dt=0.1)
            sp_julia.append(se_julia)
        f_py = ping_py.dominant_frequency(sp_py, dt=0.1, bin_ms=1.0)
        f_julia = ping_julia.dominant_frequency(sp_julia, dt=0.1, bin_ms=1.0)
        assert 30.0 <= f_py <= 80.0
        assert 30.0 <= f_julia <= 80.0
        assert abs(f_py - f_julia) < 1.5

    def test_explicit_julia_request_works(self):
        ping = PINGCircuit(
            n_excitatory=10,
            n_inhibitory=4,
            seed=1,
            backend="julia",
        )
        assert ping._use_julia is True
        ping.step(dt=0.1)

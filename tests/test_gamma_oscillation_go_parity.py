# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for PING gamma oscillation circuit (Go Parity)

import numpy as np
import pytest

from sc_neurocore.network.gamma_oscillation import (
    _HAS_GO_PING_STEP,
    PINGCircuit,
)

@pytest.mark.skipif(
    not _HAS_GO_PING_STEP,
    reason="Go kernel libgamma_oscillation.so not built",
)
class TestPythonGoParity:
    """Per-population spike rates match between the NumPy and Go
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
        ping_go = PINGCircuit(
            n_excitatory=n_e,
            n_inhibitory=n_i,
            seed=42,
            backend="go",
        )
        for _ in range(1000):
            ping_py.step(dt=0.1)
            ping_go.step(dt=0.1)
        e_py = e_go = i_py = i_go = 0
        for _ in range(5000):
            se_py, si_py = ping_py.step(dt=0.1)
            se_go, si_go = ping_go.step(dt=0.1)
            e_py += int(np.count_nonzero(se_py))
            e_go += int(np.count_nonzero(se_go))
            i_py += int(np.count_nonzero(si_py))
            i_go += int(np.count_nonzero(si_go))
        rate_e_py = e_py / n_e
        rate_e_go = e_go / n_e
        rate_i_py = i_py / n_i
        rate_i_go = i_go / n_i
        assert abs(rate_e_py - rate_e_go) / max(rate_e_py, 1e-3) < 0.10, (
            f"E rate mismatch py={rate_e_py:.3f} go={rate_e_go:.3f}"
        )
        assert abs(rate_i_py - rate_i_go) / max(rate_i_py, 1e-3) < 0.10, (
            f"I rate mismatch py={rate_i_py:.3f} go={rate_i_go:.3f}"
        )

    def test_dominant_frequency_matches_across_backends(self):
        ping_py = PINGCircuit(
            n_excitatory=400,
            n_inhibitory=100,
            seed=42,
            backend="python",
        )
        ping_go = PINGCircuit(
            n_excitatory=400,
            n_inhibitory=100,
            seed=42,
            backend="go",
        )
        for _ in range(2000):
            ping_py.step(dt=0.1)
            ping_go.step(dt=0.1)
        sp_py, sp_go = [], []
        for _ in range(8000):
            se_py, _ = ping_py.step(dt=0.1)
            sp_py.append(se_py)
            se_go, _ = ping_go.step(dt=0.1)
            sp_go.append(se_go)
        f_py = ping_py.dominant_frequency(sp_py, dt=0.1, bin_ms=1.0)
        f_go = ping_go.dominant_frequency(sp_go, dt=0.1, bin_ms=1.0)
        assert 30.0 <= f_py <= 80.0
        assert 30.0 <= f_go <= 80.0
        assert abs(f_py - f_go) < 1.5

    def test_explicit_go_request_works(self):
        ping = PINGCircuit(
            n_excitatory=10,
            n_inhibitory=4,
            seed=1,
            backend="go",
        )
        assert ping._use_go is True
        ping.step(dt=0.1)

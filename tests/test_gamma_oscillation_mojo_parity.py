# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for PING gamma oscillation circuit (Mojo Parity)

import numpy as np
import pytest

from sc_neurocore.network.gamma_oscillation import (
    _HAS_MOJO_PING_STEP,
    PINGCircuit,
)

@pytest.mark.skipif(
    not _HAS_MOJO_PING_STEP,
    reason="Mojo kernel libgamma_oscillation.so not built",
)
class TestPythonMojoParity:
    """Per-population spike rates match between the NumPy and Mojo
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
        ping_mojo = PINGCircuit(
            n_excitatory=n_e,
            n_inhibitory=n_i,
            seed=42,
            backend="mojo",
        )
        for _ in range(1000):
            ping_py.step(dt=0.1)
            ping_mojo.step(dt=0.1)
        e_py = e_mojo = i_py = i_mojo = 0
        for _ in range(5000):
            se_py, si_py = ping_py.step(dt=0.1)
            se_mojo, si_mojo = ping_mojo.step(dt=0.1)
            e_py += int(np.count_nonzero(se_py))
            e_mojo += int(np.count_nonzero(se_mojo))
            i_py += int(np.count_nonzero(si_py))
            i_mojo += int(np.count_nonzero(si_mojo))
        rate_e_py = e_py / n_e
        rate_e_mojo = e_mojo / n_e
        rate_i_py = i_py / n_i
        rate_i_mojo = i_mojo / n_i
        assert abs(rate_e_py - rate_e_mojo) / max(rate_e_py, 1e-3) < 0.10, (
            f"E rate mismatch py={rate_e_py:.3f} mojo={rate_e_mojo:.3f}"
        )
        assert abs(rate_i_py - rate_i_mojo) / max(rate_i_py, 1e-3) < 0.10, (
            f"I rate mismatch py={rate_i_py:.3f} mojo={rate_i_mojo:.3f}"
        )

    def test_dominant_frequency_matches_across_backends(self):
        ping_py = PINGCircuit(
            n_excitatory=400,
            n_inhibitory=100,
            seed=42,
            backend="python",
        )
        ping_mojo = PINGCircuit(
            n_excitatory=400,
            n_inhibitory=100,
            seed=42,
            backend="mojo",
        )
        for _ in range(2000):
            ping_py.step(dt=0.1)
            ping_mojo.step(dt=0.1)
        sp_py, sp_mojo = [], []
        for _ in range(8000):
            se_py, _ = ping_py.step(dt=0.1)
            sp_py.append(se_py)
            se_mojo, _ = ping_mojo.step(dt=0.1)
            sp_mojo.append(se_mojo)
        f_py = ping_py.dominant_frequency(sp_py, dt=0.1, bin_ms=1.0)
        f_mojo = ping_mojo.dominant_frequency(sp_mojo, dt=0.1, bin_ms=1.0)
        assert 30.0 <= f_py <= 80.0
        assert 30.0 <= f_mojo <= 80.0
        assert abs(f_py - f_mojo) < 1.5

    def test_explicit_mojo_request_works(self):
        ping = PINGCircuit(
            n_excitatory=10,
            n_inhibitory=4,
            seed=1,
            backend="mojo",
        )
        assert ping._use_mojo is True
        ping.step(dt=0.1)

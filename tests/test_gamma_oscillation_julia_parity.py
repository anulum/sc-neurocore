# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for PING gamma oscillation circuit (Julia Parity)

import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.network import gamma_oscillation
from sc_neurocore.network.gamma_oscillation import PINGCircuit


class TestPythonJuliaParity:
    """Per-population spike rates match between the NumPy and Julia
    backends within a tight tolerance.
    """

    @pytest.fixture(autouse=True)
    def julia_kernel(self) -> None:
        """Load the kernel before asking whether it is there.

        The kernel loads on first use, so a skip decided at collection read the
        flag before anything had loaded it and skipped these cases everywhere.
        """
        gamma_oscillation._ensure_julia_ping_step()
        if not gamma_oscillation._HAS_JULIA_PING_STEP:
            pytest.skip("Julia kernel not loaded")

    @pytest.mark.parametrize(
        "n_e,n_i",
        [(80, 20), (400, 100), (1000, 250)],
    )
    def test_population_rates_match(self, n_e: int, n_i: int) -> None:
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

    def test_dominant_frequency_matches_across_backends(self) -> None:
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

    def test_explicit_julia_request_works(self) -> None:
        ping = PINGCircuit(
            n_excitatory=10,
            n_inhibitory=4,
            seed=1,
            backend="julia",
        )
        assert ping._use_julia is True
        ping.step(dt=0.1)


def test_a_missing_julia_leaves_the_kernel_absent_and_says_why(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A ``None`` module entry is how the import system records an absent package.

    Julia is installed wherever this suite runs its parity cases, so its absence
    is produced by marking ``juliacall`` absent in a fresh copy of the module.
    """
    monkeypatch.setitem(sys.modules, "juliacall", None)
    spec = importlib.util.spec_from_file_location(
        "sc_neurocore_fresh_gamma_oscillation", Path(gamma_oscillation.__file__)
    )
    assert spec is not None and spec.loader is not None
    fresh = importlib.util.module_from_spec(spec)
    # Dataclasses resolve their module by name while the class is being built.
    monkeypatch.setitem(sys.modules, spec.name, fresh)
    spec.loader.exec_module(fresh)
    with caplog.at_level(logging.DEBUG, logger=fresh.__name__):
        fresh._ensure_julia_ping_step()
    assert fresh._HAS_JULIA_PING_STEP is False
    assert "Julia PING accel unavailable" in caplog.text
    with pytest.raises(RuntimeError, match="julia kernel is not available"):
        fresh.PINGCircuit(n_excitatory=8, n_inhibitory=2, seed=1, backend="julia")

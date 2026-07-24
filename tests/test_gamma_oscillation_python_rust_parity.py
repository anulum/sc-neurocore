# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPythonRustParity from former test_gamma_oscillation.py

"""Focused suite: TestPythonRustParity from former test_gamma_oscillation.py."""

from __future__ import annotations

from tests.gamma_oscillation_support import *  # noqa: F403


@pytest.mark.skipif(
    not _HAS_RUST_PING_STEP,
    reason="Rust kernel sc_neurocore_engine.py_ping_step not built",
)
class TestPythonRustParity:
    """Per-population spike rates match between the NumPy and Rust
    backends within a tight tolerance.

    Per-cell membrane voltages diverge at the float-noise level
    because NumPy may issue FMA/SIMD intermediate operations whose
    rounding differs from the Rust scalar loop. Spike-event sets
    therefore differ slightly per step once the V drift crosses the
    threshold for one cell. Aggregate population statistics (per-
    population firing rate over a 500 ms analysis window, dominant
    population FFT peak) match within a tolerance that is
    comfortably tighter than any biologically meaningful effect.
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
        ping_rs = PINGCircuit(
            n_excitatory=n_e,
            n_inhibitory=n_i,
            seed=42,
            backend="rust",
        )
        # 100 ms burn-in to settle the per-instance transient.
        for _ in range(1000):
            ping_py.step(dt=0.1)
            ping_rs.step(dt=0.1)
        # 500 ms analysis window — count total per-population spikes.
        e_py = e_rs = i_py = i_rs = 0
        for _ in range(5000):
            se_py, si_py = ping_py.step(dt=0.1)
            se_rs, si_rs = ping_rs.step(dt=0.1)
            e_py += int(np.count_nonzero(se_py))
            e_rs += int(np.count_nonzero(se_rs))
            i_py += int(np.count_nonzero(si_py))
            i_rs += int(np.count_nonzero(si_rs))
        # Rates per cell, in spikes / 500 ms / cell.
        rate_e_py = e_py / n_e
        rate_e_rs = e_rs / n_e
        rate_i_py = i_py / n_i
        rate_i_rs = i_rs / n_i
        assert abs(rate_e_py - rate_e_rs) / max(rate_e_py, 1e-3) < 0.10, (
            f"E rate mismatch py={rate_e_py:.3f} rs={rate_e_rs:.3f}"
        )
        assert abs(rate_i_py - rate_i_rs) / max(rate_i_py, 1e-3) < 0.10, (
            f"I rate mismatch py={rate_i_py:.3f} rs={rate_i_rs:.3f}"
        )

    def test_dominant_frequency_matches_across_backends(self):
        """The published 30-80 Hz peak must reproduce on both backends."""
        ping_py = PINGCircuit(
            n_excitatory=400,
            n_inhibitory=100,
            seed=42,
            backend="python",
        )
        ping_rs = PINGCircuit(
            n_excitatory=400,
            n_inhibitory=100,
            seed=42,
            backend="rust",
        )
        for _ in range(2000):  # 200 ms burn-in
            ping_py.step(dt=0.1)
            ping_rs.step(dt=0.1)
        sp_py, sp_rs = [], []
        for _ in range(8000):  # 800 ms analysis window
            se_py, _ = ping_py.step(dt=0.1)
            sp_py.append(se_py)
            se_rs, _ = ping_rs.step(dt=0.1)
            sp_rs.append(se_rs)
        f_py = ping_py.dominant_frequency(sp_py, dt=0.1, bin_ms=1.0)
        f_rs = ping_rs.dominant_frequency(sp_rs, dt=0.1, bin_ms=1.0)
        assert 30.0 <= f_py <= 80.0
        assert 30.0 <= f_rs <= 80.0
        # Both backends should land on essentially the same FFT peak;
        # 1 Hz tolerance covers spectral-bin rounding.
        assert abs(f_py - f_rs) < 1.5

    def test_explicit_rust_request_works(self):
        ping = PINGCircuit(
            n_excitatory=10,
            n_inhibitory=4,
            seed=1,
            backend="rust",
        )
        assert ping._use_rust is True
        ping.step(dt=0.1)

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="backend must be"):
            PINGCircuit(backend="haskell")

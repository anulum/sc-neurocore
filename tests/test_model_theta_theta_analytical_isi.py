# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestThetaAnalyticalISI from former test_model_theta.py

"""Focused suite: TestThetaAnalyticalISI from former test_model_theta.py."""

from __future__ import annotations

from tests.model_theta_support import *  # noqa: F403

class TestThetaAnalyticalISI:
    """ISI = π/√I (continuous time). ISI_steps = π/(√I · dt)."""

    @pytest.mark.parametrize("I", [0.5, 1.0, 2.0, 5.0])
    def test_isi_matches_analytical(self, I: float) -> None:
        """Measured ISI × dt should equal π/√I within 2%."""
        n = ThetaNeuron()
        spikes = _run(n, current=I, steps=100000)
        assert len(spikes) >= 10
        isis = np.diff(spikes[2:])
        measured_isi_time = np.mean(isis) * n.dt
        analytical_isi = np.pi / np.sqrt(I)
        rel_error = abs(measured_isi_time - analytical_isi) / analytical_isi
        assert rel_error < 0.02, (
            f"I={I}: ISI_time={measured_isi_time:.4f}, analytical={analytical_isi:.4f}, "
            f"error={rel_error:.4f}"
        )

    def test_near_constant_isi(self) -> None:
        """ISI is near-constant, with only discrete step quantisation jitter."""
        n = ThetaNeuron()
        spikes = _run(n, current=1.0, steps=50000)
        isis = np.diff(spikes[2:])
        unique_isis = np.unique(isis)
        assert len(unique_isis) <= 2, f"Too many ISI values: {unique_isis}"
        assert max(unique_isis) - min(unique_isis) <= 1, f"ISI jitter > 1: {unique_isis}"

    def test_sqrt_scaling(self) -> None:
        """f(4I)/f(I) ≈ 2 (since f ∝ √I)."""
        n1 = ThetaNeuron()
        n4 = ThetaNeuron()
        s1 = len(_run(n1, current=1.0, steps=100000))
        s4 = len(_run(n4, current=4.0, steps=100000))
        ratio = s4 / s1
        assert 1.8 < ratio < 2.2, f"f(4I)/f(I) = {ratio:.2f}, expected ~2.0"

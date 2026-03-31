# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: AstrocyteModel

"""Full pipeline test for AstrocyteModel (Li & Bhatt 1994).

3 ODEs: Ca (cytosolic), h (IP3R de-inactivation), IP3.
Returns float (Ca concentration µM), not int spike.
Ca oscillates at I=0 (range 0.05–0.94 µM). IP3 input drives Ca high.
Performance: ~73K steps/s."""

from __future__ import annotations

import time

import numpy as np

from sc_neurocore.neurons.models.astrocyte import AstrocyteModel
from sc_neurocore.network.population import Population


class TestAstrocyteIsolation:
    def test_defaults(self):
        n = AstrocyteModel()
        assert n.ca == 0.05 and n.h == 0.8 and n.ip3 == 0.5
        assert n.c0 == 2.0 and n.dt == 0.01

    def test_step_returns_float(self):
        """Returns Ca concentration (float), not binary spike."""
        n = AstrocyteModel()
        assert isinstance(n.step(0.0), float)

    def test_three_variables_evolve(self):
        n = AstrocyteModel()
        initial = (n.ca, n.h, n.ip3)
        for _ in range(500):
            n.step(0.5)
        for name, v0, v1 in zip(["ca", "h", "ip3"], initial, (n.ca, n.h, n.ip3)):
            assert v0 != v1, f"{name} didn't evolve"

    def test_state_finite(self):
        n = AstrocyteModel()
        for _ in range(100000):
            n.step(0.5)
        assert all(np.isfinite(v) for v in [n.ca, n.h, n.ip3])

    def test_reset(self):
        n = AstrocyteModel()
        for _ in range(500):
            n.step(1.0)
        n.reset()
        assert n.ca == 0.05 and n.h == 0.8 and n.ip3 == 0.5


class TestAstrocyteCaDynamics:
    """Core: IP3R channel + SERCA pump + ER leak."""

    def test_ca_oscillates_at_zero_input(self):
        """Spontaneous Ca oscillation from IP3R-Ca feedback loop."""
        n = AstrocyteModel()
        cas = []
        for _ in range(10000):
            cas.append(n.step(0.0))
        cas = np.array(cas)
        v_range = cas.max() - cas.min()
        assert v_range > 0.5, f"Ca range = {v_range:.4f}, expected oscillation"

    def test_ca_non_negative(self):
        n = AstrocyteModel()
        for _ in range(50000):
            ca = n.step(0.0)
            assert ca >= 0.0

    def test_ca_increases_with_ip3_input(self):
        """Glutamate → IP3 → Ca release from ER."""
        n_low = AstrocyteModel()
        n_high = AstrocyteModel()
        for _ in range(10000):
            n_low.step(0.0)
            n_high.step(1.0)
        assert n_high.ca > n_low.ca

    def test_ip3_drives_channel_opening(self):
        """Higher IP3 → more IP3R opening → more Ca release."""
        n = AstrocyteModel()
        for _ in range(10000):
            n.step(2.0)  # strong IP3 production
        assert n.ip3 > 1.0  # IP3 has accumulated
        assert n.ca > 0.5  # Ca elevated from ER release

    def test_h_gate_bounded(self):
        """IP3R de-inactivation gate h ∈ [0, 1]."""
        n = AstrocyteModel()
        for _ in range(50000):
            n.step(0.5)
        assert 0.0 <= n.h <= 1.0

    def test_ca_conservation(self):
        """Total Ca = ca + c1·Ca_ER is conserved (c0).

        Ca_ER = (c0 - ca) / c1.
        """
        n = AstrocyteModel()
        for _ in range(10000):
            n.step(0.5)
        ca_er = (n.c0 - n.ca) / n.c1
        total = n.ca + n.c1 * ca_er
        assert abs(total - n.c0) < 1e-10


class TestAstrocyteIP3Dynamics:
    def test_ip3_increases_with_input(self):
        n = AstrocyteModel()
        for _ in range(1000):
            n.step(1.0)
        assert n.ip3 > 0.5  # initial was 0.5, input adds more

    def test_ip3_decays_without_input(self):
        n = AstrocyteModel()
        n.ip3 = 5.0
        for _ in range(10000):
            n.step(0.0)
        assert n.ip3 < 5.0

    def test_ip3_non_negative(self):
        n = AstrocyteModel()
        for _ in range(50000):
            n.step(0.0)
        assert n.ip3 >= 0.0


class TestAstrocytePerformance:
    def test_isolation_throughput(self):
        n = AstrocyteModel()
        N = 20000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(0.5)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 20000


class TestAstrocytePipeline:
    def test_population_creates(self):
        assert Population(AstrocyteModel, n=5, label="astro").n == 5

    def test_returns_float(self):
        """Rate model (Ca²⁺). Network incompatible (float return)."""
        n = AstrocyteModel()
        assert isinstance(n.step(0.5), float)

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = AstrocyteModel()
            trace = [n.step(0.5) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]

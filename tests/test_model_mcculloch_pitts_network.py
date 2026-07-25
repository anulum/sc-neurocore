# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — McCulloch-Pitts network and analysis integration

"""Network population, monitor, and spike-analysis integration contract."""

from .model_mcculloch_pitts_support import *


def test_population_network_and_analysis_accept_integer_valued_transport() -> None:
    """The source contract remains usable through the generic Float64 network accumulator."""
    population = Population(McCullochPittsNeuron, n=8, label="mp")
    drive = PoissonInput(n=8, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
    monitor = SpikeMonitor(population)
    Network(population, drive, monitor).run(duration=0.05, dt=0.001, backend="python")
    assert monitor.count > 0

    train = np.asarray([McCullochPittsNeuron().step(1) for _ in range(100)], dtype=float)
    assert spike_count(train) == 100
    assert firing_rate(train, dt=0.001) == pytest.approx(1000.0)

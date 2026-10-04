# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rust Brunel scaling benchmark consumer

"""Run the real scaling consumer against the installed fixed-point network."""

from __future__ import annotations

import math
import tracemalloc

from tests.engine_requirement import require_engine

require_engine()
from benchmarks.scaling_benchmark import BrunelConfig, run_rust_brunel


def test_rust_scaling_consumer_returns_metrics_from_the_csr_simulator() -> None:
    """The public benchmark constructs, runs and measures the fixed-point class."""
    cfg = BrunelConfig(n_neurons=16, sim_ms=20, dt=0.1, conn_prob=0.25, seed=17)
    first = run_rust_brunel(cfg)
    second = run_rust_brunel(cfg)
    assert first is not None and second is not None
    assert first.total_spikes == second.total_spikes
    assert first.total_spikes > 0
    assert first.mean_rate_hz == first.total_spikes / (cfg.n_neurons * cfg.sim_ms / 1000)
    assert first.synaptic_events == second.synaptic_events
    assert first.activation_sparsity == second.activation_sparsity
    for value in (first.wall_time_s, first.peak_rss_mb, first.synaptic_events_per_s):
        assert math.isfinite(value) and value > 0
    assert not tracemalloc.is_tracing()

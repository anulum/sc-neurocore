# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCDenseLayer performance gate

"""Opt-in performance contract for SCDenseLayer."""

from tests.test_layers.sc_dense_layer_support import *
from tests.performance_guard import assert_load_tolerant_throughput


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_dense_layer_perf_small():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    """Benchmark a small run for basic performance sanity."""
    layer = _make_layer(n_neurons=8, length=64)
    start = time.perf_counter()
    layer.run(64)
    elapsed = time.perf_counter() - start
    assert_load_tolerant_throughput(
        label="SC dense-layer run", observed_per_second=1.0 / elapsed, strict_minimum_per_second=0.5
    )

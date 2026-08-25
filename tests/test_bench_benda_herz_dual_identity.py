# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

from pathlib import Path


def test_benchmark_sources_bind_both_identities() -> None:
    root = Path(__file__).parents[1] / "benchmarks"
    source = (root / "bench_model_benda_herz.py").read_text()
    sc = (root / "bench_model_sc_stochastic_rate_adaptation.py").read_text()
    assert "simulate_benda_herz" in source and "BendaHerzNeuron" in source
    assert (
        "simulate_sc_stochastic_rate_adaptation" in sc and "SCStochasticRateAdaptationNeuron" in sc
    )

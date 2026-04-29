# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Cross-language RK4 neuron parity harness tests

from __future__ import annotations

import pytest

from benchmarks.bench_neuron_integrators import (
    BACKEND_NAMES,
    MODEL_CASES,
    run_benchmark,
    run_parity_suite,
)


def test_cross_language_rk4_parity_harness_reports_all_backend_slots() -> None:
    suite = run_parity_suite(n_steps=1_000)

    assert suite["n_steps"] == 1_000
    assert tuple(suite["backend_order"]) == BACKEND_NAMES
    assert set(suite["models"]) == {case.model_name for case in MODEL_CASES}

    optional_backend_used = False
    for case in MODEL_CASES:
        model_result = suite["models"][case.model_name]
        assert model_result["n_steps"] == 1_000
        assert set(model_result["backends"]) == set(BACKEND_NAMES)

        python_result = model_result["backends"]["python"]
        assert python_result["used"] is True
        assert python_result["bit_exact"] is True
        assert python_result["max_abs_delta"] == 0.0

        for backend_name in BACKEND_NAMES:
            backend_result = model_result["backends"][backend_name]
            if not backend_result["used"]:
                assert backend_result["reason"]
                continue
            optional_backend_used = optional_backend_used or backend_name != "python"
            assert backend_result["n_steps"] == 1_000
            assert backend_result["spike_indices_equal"] is True
            assert backend_result["within_tolerance"] is True
            assert backend_result["max_abs_delta"] <= case.tolerance

    if not optional_backend_used:
        pytest.skip("no optional RK4 backend is available in this environment")


def test_neuron_integrator_benchmark_payload_contains_timings() -> None:
    payload = run_benchmark(n_steps=64, repeats=1)

    assert payload["meta"]["n_steps"] == 64
    assert payload["meta"]["parity_steps"] == 1_000
    assert set(payload["benchmark"]) == {case.model_name for case in MODEL_CASES}

    for case in MODEL_CASES:
        model_result = payload["benchmark"][case.model_name]
        assert set(model_result) == set(BACKEND_NAMES)
        python_timing = model_result["python"]
        assert python_timing["used"] is True
        assert python_timing["steps_per_s"] > 0
        assert python_timing["best_wall_ms"] > 0
        assert python_timing["speedup_over_python"] == 1.0

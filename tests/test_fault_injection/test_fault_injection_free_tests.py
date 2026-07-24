# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Module-level tests from former test_fault_injection.py

"""Module-level tests from former test_fault_injection.py."""

from __future__ import annotations

from fault_injection_support import *  # noqa: F403


def test_radiation_profile_rejects_non_numeric_ber():
    with pytest.raises(ValueError, match="ber must be"):
        RadiationProfile(name="bad", ber="high")  # type: ignore[arg-type]


def test_fault_injection_result_rejects_non_integer_field():
    with pytest.raises(ValueError, match="must be an integer"):
        FaultInjectionResult(
            original_popcount="x",  # type: ignore[arg-type]
            corrupted_popcount=0,
            bits_flipped=0,
            bitstream_length=10,
        )


def test_resilience_report_rejects_empty_fault_model():
    with pytest.raises(ValueError, match="fault_model must be a non-empty string"):
        ResilienceReport(**{**_VALID_REPORT, "fault_model": "   "})


def test_resilience_report_rejects_non_numeric_field():
    with pytest.raises(ValueError, match="ber must be numeric"):
        ResilienceReport(**{**_VALID_REPORT, "ber": "x"})


def test_resilience_report_rejects_non_finite_field():
    with pytest.raises(ValueError, match="mean_error must be finite"):
        ResilienceReport(**{**_VALID_REPORT, "mean_error": float("inf")})


def test_inject_rejects_non_numeric_ber():
    inj = FaultInjector(seed=0)
    with pytest.raises(ValueError, match="ber must be"):
        inj.inject(np.array([0, 1], dtype=np.uint8), FaultModel.BIT_FLIP, "x")  # type: ignore[arg-type]


def test_inject_gaussian_requires_numeric_bitstream():
    inj = FaultInjector(seed=0)
    with pytest.raises(ValueError, match="gaussian_noise requires numeric"):
        inj.inject(np.array(["a", "b"]), FaultModel.GAUSSIAN_NOISE, 0.1)


def test_inject_unsupported_fault_model_raises():
    # A FaultModel-typed object that matches none of the handled members reaches
    # the exhaustiveness guard (defended for forward compatibility / typing).
    inj = FaultInjector(seed=0)
    bogus = MagicMock(spec=FaultModel)
    with pytest.raises(ValueError, match="unsupported fault model"):
        inj.inject(np.array([0, 1], dtype=np.uint8), bogus, 0.5)


def test_inject_at_positions_rejects_non_array():
    inj = FaultInjector(seed=0)
    with pytest.raises(ValueError, match="must be a numpy.ndarray"):
        inj.inject_at_positions([0, 1, 0], [1])  # type: ignore[arg-type]


def test_inject_at_positions_rejects_non_1d():
    inj = FaultInjector(seed=0)
    with pytest.raises(ValueError, match="must be a 1-D array"):
        inj.inject_at_positions(np.zeros((2, 2), dtype=np.uint8), [0])


def test_generate_bitstream_rejects_non_numeric_probability():
    bench = ResilienceBenchmark(seed=0)
    with pytest.raises(ValueError, match="probability must be"):
        bench._generate_bitstream(8, "x")  # type: ignore[arg-type]


def test_run_rejects_non_numeric_probability():
    bench = ResilienceBenchmark(seed=0)
    with pytest.raises(ValueError, match="probability must be"):
        bench.run(fault_model=FaultModel.BIT_FLIP, ber=0.1, probability="x")  # type: ignore[arg-type]


def test_run_rejects_non_numeric_ber():
    bench = ResilienceBenchmark(seed=0)
    with pytest.raises(ValueError, match="ber must be"):
        bench.run(fault_model=FaultModel.BIT_FLIP, ber="x")  # type: ignore[arg-type]


def test_sweep_ber_rejects_non_fault_model():
    bench = ResilienceBenchmark(seed=0)
    with pytest.raises(ValueError, match="fault_model must be a FaultModel"):
        bench.sweep_ber(fault_model="bit_flip", ber_range=[0.1])  # type: ignore[arg-type]


def test_sweep_ber_rejects_non_numeric_entry():
    bench = ResilienceBenchmark(seed=0)
    with pytest.raises(ValueError, match="ber_range entries must be"):
        bench.sweep_ber(fault_model=FaultModel.BIT_FLIP, ber_range=["x"])  # type: ignore[list-item]

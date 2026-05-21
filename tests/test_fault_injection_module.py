# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Fault Injection Module Contract Tests

from __future__ import annotations

import pytest

from sc_neurocore.fault_injection.fault_injection import (
    FaultModel,
    FaultInjector,
    FaultInjectionResult,
    RadiationProfile,
    ResilienceBenchmark,
    ResilienceReport,
)


class TestRadiationProfileContracts:
    def test_presets_construct_valid_profiles(self):
        for profile in (
            RadiationProfile.terrestrial(),
            RadiationProfile.leo(),
            RadiationProfile.geo(),
            RadiationProfile.deep_space(),
        ):
            assert profile.name
            assert 0.0 <= profile.ber <= 1.0

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"name": ""}, "name"),
            ({"ber": -1e-6}, "ber"),
            ({"ber": 1.01}, "ber"),
            ({"ber": float("nan")}, "ber"),
            ({"description": 1}, "description"),
        ],
    )
    def test_rejects_invalid_contracts(self, kwargs, match):
        values = {
            "name": "LEO",
            "ber": 1e-7,
            "description": "ok",
        }
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            RadiationProfile(**values)


class TestFaultInjectionResultContracts:
    def test_probability_properties_follow_popcount_ratio(self):
        result = FaultInjectionResult(
            original_popcount=6,
            corrupted_popcount=5,
            bits_flipped=1,
            bitstream_length=10,
        )
        assert result.probability_original == 0.6
        assert result.probability_corrupted == 0.5
        assert result.absolute_error == pytest.approx(0.1)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"original_popcount": -1}, "original_popcount"),
            ({"corrupted_popcount": -1}, "corrupted_popcount"),
            ({"bits_flipped": -1}, "bits_flipped"),
            ({"bitstream_length": -1}, "bitstream_length"),
            ({"original_popcount": 11}, "original_popcount"),
            ({"corrupted_popcount": 11}, "corrupted_popcount"),
            ({"bits_flipped": 11}, "bits_flipped"),
        ],
    )
    def test_rejects_invalid_contracts(self, kwargs, match):
        values = {
            "original_popcount": 6,
            "corrupted_popcount": 5,
            "bits_flipped": 1,
            "bitstream_length": 10,
        }
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            FaultInjectionResult(**values)


class TestResilienceReportContracts:
    def test_summary_includes_core_metrics(self):
        report = ResilienceReport(
            fault_model="bit_flip",
            ber=1e-3,
            bitstream_length=128,
            num_trials=10,
            mean_error=0.01,
            std_error=0.005,
            max_error=0.03,
            p95_error=0.02,
            p99_error=0.025,
            mean_bits_flipped=1.5,
            wall_time_ms=2.5,
        )
        text = report.summary()
        assert "Fault: bit_flip" in text
        assert "Trials=10" in text

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("ber", 1.5, "ber"),
            ("bitstream_length", 0, "bitstream_length"),
            ("num_trials", 0, "num_trials"),
            ("mean_error", -0.1, "mean_error"),
            ("p95_error", 0.001, "p95_error"),
            ("p99_error", 0.019, "p99_error"),
            ("max_error", 0.024, "max_error"),
            ("mean_bits_flipped", 129.0, "mean_bits_flipped"),
            ("wall_time_ms", -1.0, "wall_time_ms"),
        ],
    )
    def test_rejects_invalid_contracts(self, field, value, match):
        values = {
            "fault_model": "bit_flip",
            "ber": 1e-3,
            "bitstream_length": 128,
            "num_trials": 10,
            "mean_error": 0.01,
            "std_error": 0.005,
            "max_error": 0.03,
            "p95_error": 0.02,
            "p99_error": 0.025,
            "mean_bits_flipped": 1.5,
            "wall_time_ms": 2.5,
        }
        values[field] = value
        with pytest.raises(ValueError, match=match):
            ResilienceReport(**values)


class TestSeedContracts:
    def test_fault_injector_reproducible_with_same_seed(self):
        bits = [0, 1, 1, 0, 1, 0, 1, 1]
        import numpy as np

        bitstream = np.array(bits, dtype=np.uint8)
        a, a_flipped = FaultInjector(seed=7).inject(bitstream, model=FaultModel.BIT_FLIP, ber=0.2)
        b, b_flipped = FaultInjector(seed=7).inject(bitstream, model=FaultModel.BIT_FLIP, ber=0.2)
        assert a_flipped == b_flipped
        assert np.array_equal(a, b)

    @pytest.mark.parametrize("seed", [1.5, "7", True])
    def test_rejects_non_integer_seed(self, seed):
        with pytest.raises(ValueError, match="seed"):
            FaultInjector(seed=seed)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="seed"):
            ResilienceBenchmark(seed=seed)  # type: ignore[arg-type]


class TestFaultInjectorInjectContracts:
    def test_rejects_invalid_inputs(self):
        import numpy as np

        injector = FaultInjector(seed=1)
        with pytest.raises(ValueError, match="numpy.ndarray"):
            injector.inject([0, 1], FaultModel.BIT_FLIP, 0.1)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="1-D"):
            injector.inject(np.zeros((2, 2), dtype=np.uint8), FaultModel.BIT_FLIP, 0.1)
        with pytest.raises(ValueError, match="non-empty"):
            injector.inject(np.zeros((0,), dtype=np.uint8), FaultModel.BIT_FLIP, 0.1)
        with pytest.raises(ValueError, match="FaultModel"):
            injector.inject(np.zeros((4,), dtype=np.uint8), "bit_flip", 0.1)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="ber"):
            injector.inject(np.zeros((4,), dtype=np.uint8), FaultModel.BIT_FLIP, 1.1)

    def test_discrete_models_reject_non_binary_streams(self):
        import numpy as np

        injector = FaultInjector(seed=1)
        bad = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        with pytest.raises(ValueError, match="binary"):
            injector.inject(bad, FaultModel.BIT_FLIP, 0.1)

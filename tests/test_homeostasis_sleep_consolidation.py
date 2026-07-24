# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSleepConsolidation from former test_homeostasis.py

"""Focused suite: TestSleepConsolidation from former test_homeostasis.py."""

from __future__ import annotations

from tests.homeostasis_support import *  # noqa: F403


class TestSleepConsolidation:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("decay_exponent", -0.1),
            ("decay_exponent", float("nan")),
            ("decay_exponent", True),
            ("noise_amplitude", -0.01),
            ("noise_amplitude", float("inf")),
            ("noise_amplitude", True),
            ("duration_fraction", 0.0),
            ("duration_fraction", 1.1),
            ("duration_fraction", float("nan")),
            ("duration_fraction", True),
        ],
    )
    def test_invalid_sleep_parameters_fail_closed(self, field: str, value: float) -> None:
        kwargs: dict[str, float] = {
            "decay_exponent": 0.5,
            "noise_amplitude": 0.01,
            "duration_fraction": 0.1,
        }
        kwargs[field] = value
        with pytest.raises(ValueError, match=field):
            SleepConsolidation(**kwargs)

    def test_apply(self) -> None:
        sleep = SleepConsolidation(decay_exponent=0.5, noise_amplitude=0.001)
        weights = [np.random.randn(10, 10)]
        consolidated = sleep.apply(weights, seed=42)
        assert len(consolidated) == 1
        assert not np.array_equal(consolidated[0], weights[0])

    def test_large_weights_decay_more(self) -> None:
        sleep = SleepConsolidation(decay_exponent=1.0, noise_amplitude=0.0)
        w = np.array([[0.1, 1.0]])
        cons = sleep.apply([w], seed=42)[0]
        # Larger weight should decay proportionally more
        ratio_before = abs(w[0, 1] / w[0, 0])
        ratio_after = abs(cons[0, 1] / max(abs(cons[0, 0]), 1e-10))
        assert ratio_after < ratio_before

    def test_should_sleep(self) -> None:
        sleep = SleepConsolidation(duration_fraction=0.1)
        assert not sleep.should_sleep(0, 100)
        assert sleep.should_sleep(10, 100)
        assert sleep.should_sleep(20, 100)
        assert not sleep.should_sleep(5, 100)

    @pytest.mark.parametrize(
        "weights",
        [
            [],
            [np.array([[1.0, float("nan")]])],
            [np.array([[1.0, float("inf")]])],
            [[1.0, 2.0]],
        ],
    )
    def test_invalid_sleep_weights_fail_closed(self, weights: object) -> None:
        sleep = SleepConsolidation()
        with pytest.raises(ValueError, match="weights"):
            sleep.apply(cast(list[npt.NDArray[np.float64]], weights))

    def test_empty_sleep_weight_matrix_fails_closed_before_reduction(self) -> None:
        sleep = SleepConsolidation()

        with pytest.raises(ValueError, match="weights must contain non-empty arrays"):
            sleep.apply([np.array([], dtype=float)])

    @pytest.mark.parametrize("seed", [True, -1, 2**32])
    def test_invalid_sleep_seed_fails_closed_before_rng(self, seed: int) -> None:
        sleep = SleepConsolidation()

        with pytest.raises(ValueError, match="seed must be an integer in"):
            sleep.apply([np.ones((2, 2))], seed=seed)

    @pytest.mark.parametrize(
        ("epoch", "total_epochs"),
        [(-1, 10), (1, 0), (1, -10), (float("nan"), 10)],
    )
    def test_invalid_sleep_schedule_fails_closed(self, epoch: int, total_epochs: int) -> None:
        sleep = SleepConsolidation()
        with pytest.raises(ValueError, match="epoch"):
            sleep.should_sleep(epoch, total_epochs)

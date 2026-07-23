# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPlasticityAndFitnessValidation from former test_validation_experiment.py

"""Focused suite: TestPlasticityAndFitnessValidation from former test_validation_experiment.py."""

from __future__ import annotations

from tests.test_bioware.validation_experiment_support import *  # noqa: F403

class TestPlasticityAndFitnessValidation:
    def test_plasticity_rejects_inverted_bounds_and_outside_state(self) -> None:
        with pytest.raises(ValueError, match="w_max_q88"):
            BiologicalSTDP(w_min_q88=2, w_max_q88=1)
        with pytest.raises(ValueError, match="configured weight bounds"):
            BiologicalSTDP().update_weight(999, 1.0)
        with pytest.raises(ValueError, match="w_max_q88"):
            BCMPlasticity(w_min_q88=2, w_max_q88=1)
        with pytest.raises(ValueError, match="configured weight bounds"):
            BCMPlasticity().update_weight(999, 1.0, 1.0)
        with pytest.raises(ValueError, match="max_threshold_q88"):
            HomeostaticPlasticity(min_threshold_q88=2, max_threshold_q88=1)
        with pytest.raises(ValueError, match="configured threshold bounds"):
            HomeostaticPlasticity().update_threshold(999, 1.0, 1.0)

    def test_fitness_rejects_negative_target_and_mutated_timestamp(self) -> None:
        with pytest.raises(ValueError, match="target_rate"):
            mea_fitness_hook([], target_rate=-1.0)
        spike = DetectedSpike(0, 0.0, -1.0)
        object.__setattr__(spike, "timestamp_s", float("inf"))
        with pytest.raises(ValueError, match="timestamps must be finite"):
            _mea_response_latency_ms(
                [spike],
                stimulus_time_s=None,
                measured_latency_ms=None,
            )

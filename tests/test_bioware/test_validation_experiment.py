# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bioware experiment-validation tests

"""Negative-path tests for experiment, audit, plasticity, and fitness APIs."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from sc_neurocore.bioware.bioware import (
    BCMPlasticity,
    BioAuditEntry,
    BioAuditLog,
    BiologicalSTDP,
    DetectedSpike,
    HomeostaticPlasticity,
    MEAConfig,
    MultiWellPlate,
    PharmModel,
    WellConfig,
    _mea_response_latency_ms,
    _quantile_indices,
    mea_fitness_hook,
)


def _entry(round_number: int = 1) -> BioAuditEntry:
    return BioAuditEntry(round_number, "2026-07-13T08:00:00+00:00", 1, 0, 1.0, 0.9)


class TestAuditValidation:
    def test_entry_rejects_invalid_metadata(self) -> None:
        with pytest.raises(ValueError, match="timestamp_iso must not be empty"):
            BioAuditEntry(1, " ", 0, 0, 0.0, 0.0)
        with pytest.raises(ValueError, match="ISO-8601"):
            BioAuditEntry(1, "not-a-time", 0, 0, 0.0, 0.0)
        with pytest.raises(ValueError, match="health_score must be <= 1"):
            BioAuditEntry(1, "2026-07-13", 0, 0, 0.0, 1.1)
        with pytest.raises(TypeError, match="notes must be a string"):
            BioAuditEntry(1, "2026-07-13", 0, 0, 0.0, 0.0, notes=cast(Any, 1))

    def test_log_rejects_invalid_identity_entries_and_order(self) -> None:
        with pytest.raises(TypeError, match="experiment_id must be a string"):
            BioAuditLog(experiment_id=cast(Any, 1))
        with pytest.raises(ValueError, match="whitespace"):
            BioAuditLog(experiment_id=" ")
        with pytest.raises(TypeError, match="BioAuditEntry"):
            BioAuditLog(entries=cast(Any, [object()]))
        with pytest.raises(ValueError, match="increase strictly"):
            BioAuditLog(entries=[_entry(2), _entry(1)])
        log = BioAuditLog(entries=[_entry(1)])
        with pytest.raises(TypeError, match="BioAuditEntry"):
            log.log(cast(Any, object()))
        with pytest.raises(ValueError, match="increase strictly"):
            log.log(_entry(1))


class TestExperimentValidation:
    def test_pharmacology_rejects_invalid_configuration_and_time_travel(self) -> None:
        with pytest.raises(ValueError, match="agent_name"):
            PharmModel(agent_name=" ")
        with pytest.raises(ValueError, match="_applied_at"):
            PharmModel(_applied_at=-2.0)
        model = PharmModel()
        model.apply(2.0)
        with pytest.raises(ValueError, match="precede"):
            model.effective_gain(1.0)

    @pytest.mark.parametrize(
        "counts",
        [
            cast(Any, [1]),
            np.ones((1, 1)),
            np.array(["x"]),
            np.array([-1.0]),
            np.array([float("nan")]),
        ],
    )
    def test_pharmacology_rejects_invalid_count_vectors(self, counts: Any) -> None:
        with pytest.raises((TypeError, ValueError)):
            PharmModel().modulate_spikes(counts, 0.0)

    def test_mutated_pharmacology_and_spikes_still_fail_closed(self) -> None:
        model = PharmModel()
        object.__setattr__(model, "gain", float("nan"))
        model.apply(0.0)
        with pytest.raises(ValueError, match="gain must be finite"):
            model.modulate_spike_events([DetectedSpike(0, 0.0, -1.0)], 0.0)

        model = PharmModel(gain=2.0, onset_delay_s=0.0)
        model.apply(0.0)
        spike = DetectedSpike(0, 0.0, -1.0)
        object.__setattr__(spike, "timestamp_s", float("inf"))
        with pytest.raises(ValueError, match="timestamps must be finite"):
            model.modulate_spike_events([spike], 0.0)

    def test_quantile_and_well_boundaries(self) -> None:
        with pytest.raises(ValueError, match="target_count must be zero"):
            _quantile_indices(0, 1)
        with pytest.raises(ValueError, match="well_id"):
            WellConfig(" ", MEAConfig())
        with pytest.raises(TypeError, match="MEAConfig"):
            WellConfig("W1", cast(Any, object()))
        with pytest.raises(ValueError, match="culture_type"):
            WellConfig("W1", MEAConfig(), culture_type=" ")

    def test_multiwell_plate_rejects_invalid_types_and_duplicates(self) -> None:
        well = WellConfig("W1", MEAConfig())
        with pytest.raises(TypeError, match="WellConfig"):
            MultiWellPlate(wells=cast(Any, [object()]))
        with pytest.raises(ValueError, match="duplicate"):
            MultiWellPlate(wells=[well, well])
        plate = MultiWellPlate([well])
        with pytest.raises(TypeError, match="WellConfig"):
            plate.add_well(cast(Any, object()))
        with pytest.raises(ValueError, match="duplicate"):
            plate.add_well(well)
        with pytest.raises(TypeError, match="MEALayout"):
            MultiWellPlate.standard_6_well(cast(Any, "60ch"))
        with pytest.raises(ValueError, match="well_id"):
            plate.get_well(" ")


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

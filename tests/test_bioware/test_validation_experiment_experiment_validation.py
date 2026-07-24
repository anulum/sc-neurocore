# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestExperimentValidation from former test_validation_experiment.py

"""Focused suite: TestExperimentValidation from former test_validation_experiment.py."""

from __future__ import annotations

from tests.test_bioware.validation_experiment_support import *  # noqa: F403


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

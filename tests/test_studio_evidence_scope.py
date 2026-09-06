# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio evidence identity scope

"""What identity each lane hands to its receipt, and what it declines to invent.

Two properties matter more than the field lists. A scope must carry the identity
that makes a wrong-model or wrong-profile pairing detectable, and it must record
nothing the payload does not actually state — a scope field guessed from an
absent block would make a chain look checked against something that was never
recorded.
"""

from __future__ import annotations

from typing import Any

import pytest

from sc_neurocore.studio.evidence_scope import (
    action_scope,
    analysis_scope,
    default_flow_attestation_dependencies,
    default_flow_attestation_scope,
    default_flow_run_scope,
    model_scan_scope,
    project_scope,
    simulation_scope,
    weight_restore_attach_dependencies,
    weight_restore_attach_scope,
    weight_restore_dependencies,
    weight_restore_scope,
)

_EXPERIMENT: dict[str, Any] = {
    "experiment": {
        "experiment_sha256": "e" * 64,
        "model": {
            "class_name": "AdExNeuron",
            "descriptor_sha256": "d" * 64,
            "module_sha256": "m" * 64,
            "schema_profile": "adex",
            "schema_sha256": "s" * 64,
        },
        "numerical": {"family": "ode", "method": "euler", "dt": 0.1},
    }
}


class TestSimulation:
    def test_the_scope_carries_the_model_and_profile_identity(self) -> None:
        assert simulation_scope(_EXPERIMENT) == {
            "descriptor_sha256": "d" * 64,
            "experiment_sha256": "e" * 64,
            "model_class": "AdExNeuron",
            "module_sha256": "m" * 64,
            "numerical_family": "ode",
            "numerical_method": "euler",
            "schema_profile": "adex",
            "schema_sha256": "s" * 64,
        }

    def test_a_result_without_an_experiment_block_records_nothing(self) -> None:
        assert simulation_scope({"states": {}}) == {}

    def test_a_field_the_payload_leaves_empty_is_not_recorded(self) -> None:
        """An empty descriptor digest is an absence, not an identity."""
        scope = simulation_scope(
            {"experiment": {"experiment_sha256": "e" * 64, "model": {"descriptor_sha256": ""}}}
        )

        assert scope == {"experiment_sha256": "e" * 64}

    def test_a_block_of_the_wrong_shape_is_skipped(self) -> None:
        assert simulation_scope({"experiment": {"model": "AdExNeuron"}}) == {}


class TestAnalysis:
    def test_the_scope_names_the_analysis_and_the_model_it_ran_on(self) -> None:
        scope = analysis_scope(
            {"analysis_metadata": {"analysis_type": "fi_curve", "input_sha256": "i" * 64}},
            {"model_name": "AdExNeuron"},
        )

        assert scope == {
            "analysis_type": "fi_curve",
            "input_sha256": "i" * 64,
            "model_class": "AdExNeuron",
        }

    def test_an_equation_analysis_names_no_model(self) -> None:
        scope = analysis_scope({"analysis_metadata": {"analysis_type": "nullclines"}}, {})

        assert scope == {"analysis_type": "nullclines"}

    def test_a_result_without_metadata_records_only_the_request(self) -> None:
        assert analysis_scope({}, {"model_name": "LIFNeuron"}) == {"model_class": "LIFNeuron"}


class TestActionsAndTraining:
    def test_an_action_is_identified_by_its_job(self) -> None:
        assert action_scope(job_id="sj_1", action_kind="studio.compile") == {
            "action_kind": "studio.compile",
            "job_id": "sj_1",
        }

    def test_a_restore_rests_on_the_training_job_that_wrote_the_checkpoint(self) -> None:
        payload = {
            "source_job_id": "sj_train",
            "materialization": {"architecture": "64->128->10", "weights_sha256": "w" * 64},
        }

        assert weight_restore_scope(payload) == {
            "architecture": "64->128->10",
            "source_job_id": "sj_train",
            "weights_sha256": "w" * 64,
        }
        dependencies = weight_restore_dependencies(payload)
        assert [dependency.to_public_dict() for dependency in dependencies] == [
            {"key": "job_id", "lane": "training", "value": "sj_train"}
        ]

    def test_an_attach_rests_on_the_restore_and_names_the_architecture(self) -> None:
        payload = {
            "source_job_id": "sj_train",
            "target_job_id": "sj_next",
            "target_architecture": "64->128->10",
        }

        assert weight_restore_attach_scope(payload) == {
            "architecture": "64->128->10",
            "source_job_id": "sj_train",
            "target_job_id": "sj_next",
        }
        dependencies = weight_restore_attach_dependencies(payload)
        assert [dependency.to_public_dict() for dependency in dependencies] == [
            {"key": "source_job_id", "lane": "training", "value": "sj_train"}
        ]

    def test_a_restore_without_a_materialisation_records_only_its_origin(self) -> None:
        assert weight_restore_scope({"source_job_id": "sj_train"}) == {"source_job_id": "sj_train"}

    @pytest.mark.parametrize(
        "reader", [weight_restore_dependencies, weight_restore_attach_dependencies]
    )
    def test_no_source_job_declares_no_input(self, reader: Any) -> None:
        """A payload that names no origin must not be given an invented one."""
        assert reader({"source_job_id": ""}) == ()


class TestGuidedFlows:
    def test_a_run_is_identified_by_its_fingerprints(self) -> None:
        payload = {
            "preset_id": "explore",
            "flow_id": "one-model",
            "reproducibility_manifest": {
                "inputs_fingerprint_sha256": "i" * 64,
                "run_fingerprint_sha256": "r" * 64,
            },
        }

        assert default_flow_run_scope(payload) == {
            "flow_id": "one-model",
            "inputs_fingerprint_sha256": "i" * 64,
            "preset_id": "explore",
            "run_fingerprint_sha256": "r" * 64,
        }

    def test_a_run_without_a_reproducibility_block_records_only_its_names(self) -> None:
        assert default_flow_run_scope({"preset_id": "explore", "flow_id": "one-model"}) == {
            "flow_id": "one-model",
            "preset_id": "explore",
        }

    def test_an_attestation_rests_on_the_run_it_attests(self) -> None:
        payload = {
            "preset_id": "explore",
            "flow_id": "one-model",
            "inputs_fingerprint_sha256": "i" * 64,
            "run_fingerprint_sha256": "r" * 64,
        }

        assert default_flow_attestation_scope(payload)["run_fingerprint_sha256"] == "r" * 64
        dependencies = default_flow_attestation_dependencies(payload)
        assert [dependency.to_public_dict() for dependency in dependencies] == [
            {"key": "run_fingerprint_sha256", "lane": "default_flow", "value": "r" * 64}
        ]

    def test_an_attestation_without_a_fingerprint_declares_no_input(self) -> None:
        assert default_flow_attestation_dependencies({"preset_id": "explore"}) == ()


class TestScansAndProjects:
    def test_a_scan_is_identified_by_its_input_and_result(self) -> None:
        payload = {"scan_metadata": {"input_sha256": "i" * 64, "result_sha256": "r" * 64}}

        assert model_scan_scope(payload) == {
            "input_sha256": "i" * 64,
            "result_sha256": "r" * 64,
        }

    def test_a_scan_without_metadata_records_nothing(self) -> None:
        assert model_scan_scope({"schema_version": "studio.model-scan.v1"}) == {}

    def test_a_project_is_identified_by_name_and_version(self) -> None:
        assert project_scope({"name": "demo", "version": "3.16.0"}) == {
            "project_name": "demo",
            "project_version": "3.16.0",
        }

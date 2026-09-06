# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio evidence identity scope and declared inputs

"""What each Studio evidence payload was produced under, and what it rests on.

A receipt is only as useful as the identity it carries. These readers take the
identity a payload already records — the experiment digest, the model class and
its descriptor and schema digests, the numerical profile, the job that produced
it — and express it as the flat scope a receipt holds, so two lanes that record
the same fact under different key names still compare.

Dependencies are read the same way, and only where a derivation genuinely
exists. A restored checkpoint rests on the training job that wrote it; an
attached checkpoint rests on the restore; a guided-flow attestation rests on the
run it attests. A simulation rests on nothing else in a pack, so it declares
nothing: an invented edge would make a chain look checked where it was not.
"""

from __future__ import annotations

from collections.abc import Mapping

from sc_neurocore.studio.evidence_receipt import EvidenceDependency

#: Model identity fields copied out of an experiment block, source key first.
_MODEL_FIELDS = (
    ("class_name", "model_class"),
    ("descriptor_sha256", "descriptor_sha256"),
    ("module_sha256", "module_sha256"),
    ("schema_profile", "schema_profile"),
    ("schema_sha256", "schema_sha256"),
)

#: Numerical profile fields copied out of an experiment block.
_NUMERICAL_FIELDS = (("family", "numerical_family"), ("method", "numerical_method"))


def simulation_scope(payload: Mapping[str, object]) -> dict[str, str]:
    """Return the identity a simulation result was produced under.

    Parameters
    ----------
    payload : mapping
        A Studio simulation result carrying an ``experiment`` block.

    Returns
    -------
    dict of str to str
        Experiment digest, model identity and numerical profile, omitting any
        field the payload does not record.
    """
    experiment = payload.get("experiment")
    if not isinstance(experiment, Mapping):
        return {}
    scope = _text_field(experiment, "experiment_sha256", "experiment_sha256")
    scope.update(_block_fields(experiment.get("model"), _MODEL_FIELDS))
    scope.update(_block_fields(experiment.get("numerical"), _NUMERICAL_FIELDS))
    return scope


def analysis_scope(
    payload: Mapping[str, object], request_payload: Mapping[str, object]
) -> dict[str, str]:
    """Return the identity an analysis result was produced under.

    Parameters
    ----------
    payload : mapping
        The analysis result, carrying ``analysis_metadata``.
    request_payload : mapping
        The request the analysis ran, which names the model when there is one.

    Returns
    -------
    dict of str to str
        Analysis type, input digest and model class where the request named one.
    """
    scope: dict[str, str] = {}
    metadata = payload.get("analysis_metadata")
    if isinstance(metadata, Mapping):
        scope.update(_text_field(metadata, "analysis_type", "analysis_type"))
        scope.update(_text_field(metadata, "input_sha256", "input_sha256"))
    scope.update(_text_field(request_payload, "model_name", "model_class"))
    return scope


def action_scope(*, job_id: str, action_kind: str) -> dict[str, str]:
    """Return the identity of one worker-backed Studio action.

    Parameters
    ----------
    job_id : str
        Job that executed the action; what dependent evidence resolves against.
    action_kind : str
        Stable action identifier, such as ``studio.compile``.

    Returns
    -------
    dict of str to str
        The job and action identity.
    """
    return {"action_kind": action_kind, "job_id": job_id}


def weight_restore_scope(payload: Mapping[str, object]) -> dict[str, str]:
    """Return the identity of a materialised training checkpoint.

    The architecture and the weight digest live in the materialisation block,
    and they are what an attach has to agree with: attaching a checkpoint to a
    network of a different shape is the wrong-model case for this lane.
    """
    scope = _text_field(payload, "source_job_id", "source_job_id")
    materialization = payload.get("materialization")
    if isinstance(materialization, Mapping):
        scope.update(_text_field(materialization, "architecture", "architecture"))
        scope.update(_text_field(materialization, "weights_sha256", "weights_sha256"))
    return scope


def weight_restore_dependencies(
    payload: Mapping[str, object],
) -> tuple[EvidenceDependency, ...]:
    """Return the training job a materialised checkpoint rests on."""
    source_job_id = payload.get("source_job_id")
    if not isinstance(source_job_id, str) or not source_job_id:
        return ()
    return (EvidenceDependency(lane="training", key="job_id", value=source_job_id),)


def weight_restore_attach_scope(payload: Mapping[str, object]) -> dict[str, str]:
    """Return the identity of a checkpoint attached to a new training run."""
    scope = _text_field(payload, "source_job_id", "source_job_id")
    scope.update(_text_field(payload, "target_job_id", "target_job_id"))
    scope.update(_text_field(payload, "target_architecture", "architecture"))
    return scope


def weight_restore_attach_dependencies(
    payload: Mapping[str, object],
) -> tuple[EvidenceDependency, ...]:
    """Return the materialised checkpoint an attach rests on."""
    source_job_id = payload.get("source_job_id")
    if not isinstance(source_job_id, str) or not source_job_id:
        return ()
    return (EvidenceDependency(lane="training", key="source_job_id", value=source_job_id),)


def default_flow_run_scope(payload: Mapping[str, object]) -> dict[str, str]:
    """Return the identity of one guided default-flow run."""
    scope = _text_field(payload, "preset_id", "preset_id")
    scope.update(_text_field(payload, "flow_id", "flow_id"))
    reproducibility = payload.get("reproducibility_manifest")
    if isinstance(reproducibility, Mapping):
        scope.update(
            _text_field(reproducibility, "run_fingerprint_sha256", "run_fingerprint_sha256")
        )
        scope.update(
            _text_field(reproducibility, "inputs_fingerprint_sha256", "inputs_fingerprint_sha256")
        )
    return scope


def default_flow_attestation_scope(payload: Mapping[str, object]) -> dict[str, str]:
    """Return the identity of one guided default-flow attestation."""
    scope = _text_field(payload, "preset_id", "preset_id")
    scope.update(_text_field(payload, "flow_id", "flow_id"))
    scope.update(_text_field(payload, "run_fingerprint_sha256", "run_fingerprint_sha256"))
    scope.update(_text_field(payload, "inputs_fingerprint_sha256", "inputs_fingerprint_sha256"))
    return scope


def default_flow_attestation_dependencies(
    payload: Mapping[str, object],
) -> tuple[EvidenceDependency, ...]:
    """Return the guided-flow run an attestation rests on."""
    fingerprint = payload.get("run_fingerprint_sha256")
    if not isinstance(fingerprint, str) or not fingerprint:
        return ()
    return (
        EvidenceDependency(lane="default_flow", key="run_fingerprint_sha256", value=fingerprint),
    )


def model_scan_scope(payload: Mapping[str, object]) -> dict[str, str]:
    """Return the identity of one catalogue scan."""
    metadata = payload.get("scan_metadata")
    if not isinstance(metadata, Mapping):
        return {}
    scope = _text_field(metadata, "input_sha256", "input_sha256")
    scope.update(_text_field(metadata, "result_sha256", "result_sha256"))
    return scope


def project_scope(payload: Mapping[str, object]) -> dict[str, str]:
    """Return the identity of one saved project workspace."""
    scope = _text_field(payload, "name", "project_name")
    scope.update(_text_field(payload, "version", "project_version"))
    return scope


def _text_field(block: Mapping[str, object], source: str, field: str) -> dict[str, str]:
    value = block.get(source)
    if isinstance(value, str) and value:
        return {field: value}
    return {}


def _block_fields(block: object, fields: tuple[tuple[str, str], ...]) -> dict[str, str]:
    if not isinstance(block, Mapping):
        return {}
    scope: dict[str, str] = {}
    for source, field in fields:
        scope.update(_text_field(block, source, field))
    return scope


__all__ = [
    "action_scope",
    "analysis_scope",
    "default_flow_attestation_dependencies",
    "default_flow_attestation_scope",
    "default_flow_run_scope",
    "model_scan_scope",
    "project_scope",
    "simulation_scope",
    "weight_restore_attach_dependencies",
    "weight_restore_attach_scope",
    "weight_restore_dependencies",
    "weight_restore_scope",
]

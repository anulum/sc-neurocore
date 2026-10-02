# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Worker job refusal codes

"""Render known job-policy reasons without trusting worker-supplied error text.

Workers may select a finite code; its public wording belongs to this source.
Unknown codes, legacy output and arbitrary text never acquire authored status.
Cross-domain failures outside this job-policy vocabulary use the fixed fallback
while their diagnostic text remains in custody.
"""

from sc_neurocore.refusals import AuthoredRefusal
from sc_neurocore.studio.platform.jobs_failures import (
    GENERIC_JOB_FAILURE,
    StudioJobError,
)

WORKER_FAILURE_SCHEMA = "studio.worker.failure.v1"
_MESSAGES = {
    "artifact_size": "Studio job artifact exceeds configured size limit.",
    "artifact_absent": "Studio job artifact is unavailable.",
    "event_json": "Studio job event payload must be JSON.",
    "event_size": "Studio job event log exceeds configured size limit.",
    "seed_absent": "Studio job seed input is unavailable.",
    "seed_size": "Studio job seed input exceeds configured size limit.",
    "control_seed_absent": "Studio job control seed is unavailable.",
    "control_seed_size": "Studio job control seed exceeds configured size limit.",
    "control_size": "Studio job control command exceeds configured size limit.",
    "control_json": "Studio job control command is not valid JSON.",
    "control_object": "Studio job control command must be a JSON object.",
    "artifact_path": "Studio job artifact path escapes the job directory.",
    "seed_path": "Studio job seed-input path escapes the seed directory.",
    "control_seed_path": "Studio job control-seed path escapes the control-seed directory.",
    "job_path": "Studio job path escapes the job root.",
    "relative_path": "Path must be a confined relative path.",
}
_CODES = {message: code for code, message in _MESSAGES.items()}


def worker_refusal_code(error: BaseException) -> str | None:
    """Encode only a typed authored refusal in the reviewed job-policy vocabulary.

    Parameters
    ----------
    error : BaseException
        Source-produced exception, whose type and exact authored wording are checked.

    Returns
    -------
    str or None
        Finite refusal code, or no code for an unmarked or cross-domain fault.
    """
    return _CODES.get(str(error)) if isinstance(error, AuthoredRefusal) else None


def worker_job_error(payload: dict[object, object], *, diagnostic: str) -> StudioJobError:
    """Retain worker diagnostics and render a known code from the versioned contract.

    A code conveys a worker's reported reason, not proof of the incident. Public
    text is always a source-owned constant, even if the worker forges the code.
    A worker ``public_error`` field is deliberately ignored.

    Parameters
    ----------
    payload : dict
        Untrusted worker fields. Only the expected schema and a known code select
        source-owned wording; their presence does not prove the incident.
    diagnostic : str
        Retained original diagnostic, absent from the public projection.

    Returns
    -------
    StudioJobError
        Private diagnostic paired with finite source wording or the fixed fallback.
    """
    code = payload.get("refusal_code")
    message = GENERIC_JOB_FAILURE
    if payload.get("failure_schema") == WORKER_FAILURE_SCHEMA and isinstance(code, str):
        message = _MESSAGES.get(code, GENERIC_JOB_FAILURE)
    return StudioJobError(diagnostic, public_message=message)

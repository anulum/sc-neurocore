# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native compiler response decoding

"""Validate untrusted process results before projecting compiler HTTP responses."""

from __future__ import annotations

from collections.abc import Mapping

from fastapi import HTTPException
from pydantic import ValidationError

from sc_neurocore.studio.platform.equation_ir_process import (
    EquationIRWorkerRefusal,
    EquationIRWorkerResult,
)


def decode_equation_ir_response(response: Mapping[str, object]) -> dict[str, object]:
    """Decode a completed process result and project its public response.

    Parameters
    ----------
    response : Mapping of str to object
        JSON-decoded process envelope with the runtime's optional job receipt.
        Completion of a process does not establish validity of its domain result.

    Returns
    -------
    dict of str to object
        Validated compiler result and the original runtime receipt.

    Raises
    ------
    HTTPException
        Sanitized status 500 for a malformed worker envelope, or the validated
        controlled refusal status with its reason and original job receipt.
    """
    receipt = response.get("studio_job_receipt")
    try:
        outcome = EquationIRWorkerResult.model_validate(
            {key: value for key, value in response.items() if key != "studio_job_receipt"}
        ).root
    except ValidationError as exc:
        raise HTTPException(500, "studio_compiler_result_invalid") from exc
    if isinstance(outcome, EquationIRWorkerRefusal):
        raise HTTPException(
            outcome.refusal.status_code,
            {**outcome.refusal.model_dump(), "studio_job_receipt": receipt},
        )
    return {**outcome.result.model_dump(mode="json"), "studio_job_receipt": receipt}

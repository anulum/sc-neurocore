# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Named catalogue scan worker

"""Run the existing complete catalogue scan outside the API interpreter."""

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, FiniteFloat, PositiveFloat

from sc_neurocore.studio.model_scan import scan_all_models
from sc_neurocore.studio.platform.jobs_context import StudioJobContext


class _ScanRequest(BaseModel):
    """Bounded-shape worker envelope; values retain the existing model units."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    current: FiniteFloat
    duration: PositiveFloat


def execute_model_scan_process_task(
    context: StudioJobContext, payload: Mapping[str, object]
) -> dict[str, object]:
    """Validate the operating point and classify every model in the catalogue.

    Parameters
    ----------
    context : StudioJobContext
        Registered worker context. Process cancellation is enforced by the
        supervisor; the scan also retains its cooperative stop callback.
    payload : mapping
        Exactly ``current`` (finite, model-native input units) and ``duration``
        (finite positive milliseconds). The HTTP route supplies these values.

    Returns
    -------
    dict[str, object]
        Complete ``studio.model-scan.v1`` response with configuration/result
        digests and explicit per-model failures. No catalogue subset is used.

    Raises
    ------
    ValueError
        The envelope has missing/extra fields or invalid numeric values.
    StudioJobCancelled
        The cooperative callback requests cancellation before a model runs.
    """
    request = _ScanRequest.model_validate(dict(payload))
    return dict(
        scan_all_models(
            current=request.current,
            duration=request.duration,
            should_stop=lambda: context.cancelled,
        )
    )

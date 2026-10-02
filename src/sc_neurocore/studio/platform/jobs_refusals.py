# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio job refusal boundaries

"""Preserve typed authored refusals without promoting generated exception text."""

from sc_neurocore.refusals import AuthoredRefusal
from sc_neurocore.studio.platform.jobs_models import (
    StudioJobArtifactRefused,
    StudioJobRefused,
)


def job_refusal(error: ValueError, *, fallback: str) -> StudioJobRefused:
    """Translate a validation fault, preserving only explicitly authored messages.

    Parameters
    ----------
    error : ValueError
        Original validation fault; its text is public only for AuthoredRefusal.
    fallback : str
        Source-owned job refusal used for every unmarked fault.

    Returns
    -------
    StudioJobRefused
        Caller-facing refusal; the raising caller retains ``error`` as its cause.
    """
    return StudioJobRefused(str(error) if isinstance(error, AuthoredRefusal) else fallback)


def artifact_refusal(error: ValueError, *, fallback: str) -> StudioJobArtifactRefused:
    """Translate artifact validation with the same explicit provenance boundary.

    Parameters
    ----------
    error : ValueError
        Original path-validation fault whose authored type may preserve its text.
    fallback : str
        Source-owned artifact reason used for an unmarked validation fault.

    Returns
    -------
    StudioJobArtifactRefused
        Authored refusal compatible with existing artifact-unavailable handlers.
    """
    return StudioJobArtifactRefused(str(error) if isinstance(error, AuthoredRefusal) else fallback)

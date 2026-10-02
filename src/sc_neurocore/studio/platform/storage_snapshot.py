# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Qualified storage public job snapshots

"""Retain the public projection after a current, peer-verified storage exchange."""

from dataclasses import replace

from sc_neurocore.studio.platform.jobs_failures import StudioJobError
from sc_neurocore.studio.platform.jobs_models import StudioJobRecord


def storage_job_record(record: StudioJobRecord) -> StudioJobRecord:
    """Qualify the storage authority's public error without inventing diagnostics.

    Call only after OS peer verification and complete validation of record v3,
    query v2, cancel v2 or purge v2. Those producers export qualified public
    messages; the old versions could export raw diagnostics and are refused.
    This projection does not mark serialized text as an AuthoredRefusal or
    authenticate a record. Generic snapshot and legacy replay decoding remain
    unqualified. The authority retains diagnostics; they never enter this view.

    Parameters
    ----------
    record : StudioJobRecord
        Complete decoded and correlated snapshot from the verified storage peer.

    Returns
    -------
    StudioJobRecord
        Snapshot retaining the authority's public message without an authored
        exception marker. Its error contains no additional private diagnostic.
    """
    if record.error is None:
        return record
    return replace(record, error=StudioJobError(record.error, public_message=record.error))

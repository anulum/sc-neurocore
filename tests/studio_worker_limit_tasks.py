# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real Studio worker limit tasks

"""Import-stable tasks that inspect and exercise kernel worker limits."""

from __future__ import annotations

import resource
from collections.abc import Mapping

from sc_neurocore.studio.platform.jobs import StudioJobContext

_IMPORT_LIMITS = {
    name: resource.getrlimit(getattr(resource, name))[0]
    for name in ("RLIMIT_DATA", "RLIMIT_CPU", "RLIMIT_NOFILE", "RLIMIT_FSIZE", "RLIMIT_CORE")
}


def report_import_limits(
    context: StudioJobContext, payload: Mapping[str, object]
) -> dict[str, object]:
    """Return limits captured when the task module was first imported."""
    del context, payload
    return dict(_IMPORT_LIMITS)


def allocate_beyond_data_limit(
    context: StudioJobContext, payload: Mapping[str, object]
) -> dict[str, object]:
    """Allocate the requested bytes through the real kernel limited process."""
    del context
    size = payload.get("bytes")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError("Allocation size must be a positive integer.")
    allocated = bytearray(size)
    return {"allocated_bytes": len(allocated)}

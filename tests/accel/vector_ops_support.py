# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared vector-operations test support

"""Performance-gate support for vectorized bitstream operations."""

import os


def _perf_enabled() -> bool:
    return os.environ.get("SC_NEUROCORE_PERF") == "1"

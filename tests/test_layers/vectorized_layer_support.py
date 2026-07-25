# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared VectorizedSCLayer test calculations

"""Small shared calculations for focused VectorizedSCLayer test suites."""

import os


def _perf_enabled() -> bool:
    """Return whether opt-in performance checks are enabled."""
    return os.environ.get("SC_NEUROCORE_PERF") == "1"


def _expected_words(length: int) -> int:
    """Return the packed uint64 word count for a bitstream length."""
    return (length + 63) // 64

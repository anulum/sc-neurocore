# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Evidence export input budget

"""Explicit aggregate input budget, independent of per-artifact write limits."""

DEFAULT_EVIDENCE_MAX_INPUT_BYTES = 256 * 1024 * 1024


def validate_evidence_input_limit(value: int) -> int:
    """Return a positive integer byte budget; reject booleans and coercions.

    Parameters
    ----------
    value : int
        Aggregate encoded-metadata and binary-seed ceiling, in bytes.

    Returns
    -------
    int
        Unchanged validated byte limit; no settings or filesystem are modified.

    Raises
    ------
    ValueError
        The value is not a positive integer, including boolean inputs.
    """
    if type(value) is not int or value <= 0:
        raise ValueError("Studio evidence input byte limit must be a positive integer.")
    return value


def parse_evidence_input_limit(value: str | None) -> int:
    """Parse the environment override, preserving the default only when absent.

    Parameters
    ----------
    value : str or None
        Explicit byte count, or None for the 256 MiB operational default.

    Returns
    -------
    int
        Validated aggregate input limit, independent of per-artifact limits.

    Raises
    ------
    ValueError
        An explicit value is empty, non-integral or non-positive.
    """
    if value is None:
        return DEFAULT_EVIDENCE_MAX_INPUT_BYTES
    return validate_evidence_input_limit(int(value))

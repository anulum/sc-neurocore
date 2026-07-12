# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared SystemVerilog emission contracts

"""Shared constants and identifier validation for chiplet RTL emitters."""

from __future__ import annotations

import re


SPDX_HEADER = """\
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li"""

_SV_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _require_sv_identifier(value: str, field_name: str) -> str:
    """Validate a SystemVerilog identifier fragment.

    Parameters
    ----------
    value
        Candidate identifier fragment.
    field_name
        Public field name used in the validation error.

    Returns
    -------
    str
        The validated value.

    Raises
    ------
    ValueError
        If ``value`` is not a legal SystemVerilog identifier.
    """
    if not _SV_IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a valid SystemVerilog identifier")
    return value

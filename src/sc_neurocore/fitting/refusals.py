# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Authored laboratory validation refusals

"""Caller-facing laboratory messages shared by fitting and HTTP admission."""

from sc_neurocore.refusals import AuthoredRefusal


class LaboratoryRefusal(AuthoredRefusal):
    """A deliberate laboratory validation message safe for the caller to read.

    Generated conversion, schema and structural exceptions remain unmarked;
    the HTTP boundary replaces them with its fixed malformed-document message.
    """

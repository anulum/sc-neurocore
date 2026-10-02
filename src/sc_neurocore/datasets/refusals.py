# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Authored dataset input refusals

"""Identify deliberate dataset, split and encoder validation messages."""

from sc_neurocore.refusals import AuthoredRefusal


class DatasetRefusal(AuthoredRefusal):
    """A dataset validation message written for callers.

    This remains a ValueError for library callers. Generated conversion and
    decoder faults keep their original types; consumers must not promote their
    text to this marker.
    """

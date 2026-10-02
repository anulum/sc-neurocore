# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Authored training lifecycle refusals

"""Mark deliberate training admission and weight lifecycle refusals."""

from sc_neurocore.refusals import AuthoredRefusal


class TrainingRefusal(AuthoredRefusal):
    """A training validation message that may cross the HTTP boundary.

    Producers raise this only with source-owned messages. Generated parser,
    conversion and worker diagnostic text must not be used as the marker's
    message. Existing ValueError handlers remain compatible.
    """

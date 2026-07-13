# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio policy historical test-path sentinel

"""Historical test-path sentinel for the modular Studio policy suite."""

from sc_neurocore.studio.platform import policy


def test_studio_policy_facade_has_unique_explicit_exports() -> None:
    """The historical facade exposes a deterministic duplicate-free contract."""

    assert policy.__all__ == sorted(policy.__all__)
    assert len(policy.__all__) == len(set(policy.__all__))

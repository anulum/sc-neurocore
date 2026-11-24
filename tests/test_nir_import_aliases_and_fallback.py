# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAliasesAndFallback from former test_nir_import.py

"""Focused suite: TestAliasesAndFallback from former test_nir_import.py."""

from __future__ import annotations

import pytest
from tests.nir_import_support import (
    _one,
)


class TestAliasesAndFallback:
    """Node-type aliases and the refusal of unknown types."""

    @pytest.mark.parametrize(
        "alias,canonical",
        [
            ("LeakyIntegrateAndFire", "lif"),
            ("integrate_and_fire", "if"),
            ("Leaky Integrator", "li"),
            ("cuba-lif", "cuba_lif"),
            ("CUBALI", "cuba_li"),
            ("Integrator", "integrator"),
        ],
    )
    def test_alias_resolution(self, alias: str, canonical: str) -> None:
        """Each alias resolves to its canonical node type."""
        assert _one(alias).node_types["n0"] == canonical

    def test_unknown_type_is_refused_rather_than_replaced(self) -> None:
        """An unknown node type is refused, with the supported types listed."""
        # Substituting a leaky integrator would silently change the network.
        with pytest.raises(ValueError, match=r"unsupported type 'Mystery'.*supported: cuba_li"):
            _one("Mystery", tau=5.0)

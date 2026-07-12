# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared CLI test invocation helpers

"""Invoke the public CLI and construct optional-dependency module fixtures."""

from __future__ import annotations

import types
from collections.abc import Sequence

from sc_neurocore.cli import main


def run_cli(*argv: str) -> int:
    """Run the public CLI with an explicit argument vector.

    Parameters
    ----------
    *argv : str
        Arguments following the executable name.

    Returns
    -------
    int
        Command exit status.
    """
    args: Sequence[str] = argv
    return main(args)


def fake_module(name: str, **attributes: object) -> types.ModuleType:
    """Build a module fixture for one optional dependency boundary.

    Parameters
    ----------
    name : str
        Import name represented by the fixture.
    **attributes : object
        Attributes exposed by the module.

    Returns
    -------
    types.ModuleType
        Module object suitable for a temporary ``sys.modules`` entry.
    """
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Clear refusal when a feature needs a source-checkout-only package

"""Refuse clearly when a shipped feature needs a package the wheel leaves out.

``pyproject.toml`` keeps some research packages out of the wheel, as
``docs/architecture/package_boundary_decision.md`` records. A shipped module
that needs one of them imports it where it is used, after
:func:`require_source_checkout_package`, so the shipped module itself stays
importable from an installed wheel and the feature that needs the package
names the reason when it is called.
"""

from __future__ import annotations

import importlib


def require_source_checkout_package(package: str, feature: str) -> None:
    """Import ``package``, or refuse because only a source checkout ships it.

    Parameters
    ----------
    package : str
        Fully qualified name of a package that the wheel does not ship.
    feature : str
        The feature that needs ``package``; the refusal names it.

    Raises
    ------
    ModuleNotFoundError
        ``package`` itself, or a parent package of it, cannot be found, as in an
        installed wheel. A missing third-party dependency inside ``package`` is
        raised unchanged, because a source checkout would not fix it.
    """
    try:
        importlib.import_module(package)
    except ModuleNotFoundError as exc:
        missing = exc.name
        if missing is None or not (missing == package or package.startswith(missing + ".")):
            raise
        raise ModuleNotFoundError(
            f"{feature} needs {package}, which ships only in a source checkout of "
            "SC-NeuroCore, not in the wheel.",
            name=package,
        ) from exc

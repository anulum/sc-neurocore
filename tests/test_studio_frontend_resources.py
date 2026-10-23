# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio interface discovery contracts

"""Keep installed UI precedence and checkout layouts independent of API imports."""

from __future__ import annotations

from pathlib import Path

import sc_neurocore.studio
from sc_neurocore.studio.frontend_resources import (
    studio_frontend_candidates,
    studio_frontend_dir,
)


def test_discovery_retains_packaged_and_checkout_locations(tmp_path: Path) -> None:
    """Keep both source layouts behind the selected installation's own assets.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Parent of the application module's source-layout path.
    """
    app = tmp_path / "checkout/src/sc_neurocore/studio/app.py"
    candidates = studio_frontend_candidates(str(app))
    assert candidates == (
        Path(sc_neurocore.studio.__file__).resolve().parent / "frontend_dist",
        tmp_path / "checkout/studio/frontend/dist",
        tmp_path / "studio/frontend/dist",
    )


def test_discovery_uses_the_first_built_entry_and_refuses_incomplete_assets(tmp_path: Path) -> None:
    """Ignore absent entries and entry directories while retaining candidate order.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Actual filesystem directories representing UI build candidates.
    """
    missing = tmp_path / "missing"
    incomplete = tmp_path / "incomplete"
    (incomplete / "index.html").mkdir(parents=True)
    first, second = tmp_path / "first", tmp_path / "second"
    for location in (first, second):
        location.mkdir()
        (location / "index.html").write_text("<!doctype html><title>Studio</title>")
    assert studio_frontend_dir((missing, incomplete)) is None
    assert studio_frontend_dir(()) is None
    assert studio_frontend_dir((missing, incomplete, first, second)) == first
    assert studio_frontend_dir((second, first)) == second

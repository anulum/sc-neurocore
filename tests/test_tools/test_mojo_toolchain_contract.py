# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — CI Mojo compiler version contract

"""Keep the editable CI install on the same Mojo version as the build toolchain."""

from __future__ import annotations

import sys
from pathlib import Path

from packaging.requirements import Requirement

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[2]
LOCKED_INPUTS = (
    "ci-annealing.txt",
    "ci-dev.txt",
    "ci-mpi.txt",
    "ci-optics.txt",
    "workstation.txt",
)


def test_editable_install_cannot_shadow_pixi_with_another_mojo_version() -> None:
    """Bind the dev extra and all CI locks to the compiler used by Pixi builds."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    pixi = tomllib.loads((ROOT / "src/sc_neurocore/accel/mojo/pixi.toml").read_text())
    expected = pixi["dependencies"]["mojo"]
    assert expected.startswith("==")

    dev_mojo = [
        Requirement(raw)
        for raw in project["project"]["optional-dependencies"]["dev"]
        if Requirement(raw).name == "mojo"
    ]
    assert len(dev_mojo) == 1
    assert str(dev_mojo[0].specifier) == expected
    assert dev_mojo[0].marker is not None
    assert dev_mojo[0].marker.evaluate({"platform_system": "Linux"})

    for name in LOCKED_INPUTS:
        lines = (ROOT / "requirements" / name).read_text().splitlines()
        locked = [Requirement(line.rstrip(" \\")) for line in lines if line.startswith("mojo==")]
        assert len(locked) == 1, name
        assert str(locked[0].specifier) == expected, name

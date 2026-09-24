# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Which execution lanes a distribution provides

"""What each execution lane needs, and whether this installation has it.

The published wheel is pure Python. It always carries the Python reference
implementations; the Rust engine is a separate optional package; the Julia,
Go and Mojo lanes run only from a source checkout, because the wheel ships
neither their kernel sources nor libraries built from them. A lane's
availability is therefore read from the installation itself -- the importable
packages and the files beside this module -- never inferred from a checkout
the code happens to sit next to. A lane whose resources are present can still
refuse an individual kernel: each kernel checks its own artefact before use.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from typing import Literal

Lane = Literal["python", "rust", "julia", "go", "mojo"]
Distribution = Literal["bundled", "optional-package", "source-checkout"]

ACCEL_ROOT = Path(__file__).resolve().parent / "accel"
"""The installed ``sc_neurocore.accel`` directory, read as files: importing the
package would load its numerical dependencies just to report on them."""


@dataclass(frozen=True, slots=True)
class LaneContract:
    """How a distribution provides one execution lane.

    Parameters
    ----------
    lane:
        The lane's name, as the kernels' ``backend`` arguments use it.
    distribution:
        ``bundled`` ships in every distribution, ``optional-package`` needs a
        separately installed package, ``source-checkout`` needs a checkout.
    requirement:
        What an installation needs for the lane, in words a user can act on.
    """

    lane: Lane
    distribution: Distribution
    requirement: str


@dataclass(frozen=True, slots=True)
class LaneStatus:
    """Whether this installation has one lane's resources.

    Parameters
    ----------
    lane:
        The lane's name.
    distribution:
        How distributions provide it.
    resources_present:
        Whether the packages and files the lane needs are present here.
    detail:
        What was found, or what is missing and how to provide it.
    """

    lane: Lane
    distribution: Distribution
    resources_present: bool
    detail: str


CONTRACT: tuple[LaneContract, ...] = (
    LaneContract(
        "python",
        "bundled",
        "Always available: the Python reference implementations ship in every distribution.",
    ),
    LaneContract(
        "rust",
        "optional-package",
        "Install the sc_neurocore_engine package; the pure-Python wheel does not "
        "contain the compiled engine.",
    ),
    LaneContract(
        "julia",
        "source-checkout",
        "Run from a source checkout with juliacall installed (the `julia` extra): "
        "the wheel does not ship the Julia kernel sources.",
    ),
    LaneContract(
        "go",
        "source-checkout",
        "Build the Go shared libraries in a source checkout; the wheel ships neither "
        "the Go sources nor built libraries.",
    ),
    LaneContract(
        "mojo",
        "source-checkout",
        "Build the Mojo shared libraries with the Mojo toolchain in a source checkout; "
        "the wheel ships neither the Mojo sources nor built libraries.",
    ),
)


def _importable(module: str) -> bool:
    # ``None`` in ``sys.modules`` blocks an import, and a module loaded without
    # a spec is still loaded; answering from ``sys.modules`` first keeps
    # ``find_spec`` from raising on either. Only top-level names are probed.
    if module in sys.modules:
        return sys.modules[module] is not None
    return importlib.util.find_spec(module) is not None


def _has_files(directory: Path, pattern: str) -> bool:
    return directory.is_dir() and any(path.is_file() for path in directory.rglob(pattern))


def _present(lane: Lane, accel_root: Path) -> tuple[bool, str]:
    if lane == "python":
        return True, "Python reference implementations are part of this package."
    if lane == "rust":
        if _importable("sc_neurocore_engine"):
            return True, "The sc_neurocore_engine package is installed."
        return False, "The sc_neurocore_engine package is not installed."
    if lane == "julia":
        kernels = _has_files(accel_root / "julia", "*.jl")
        if kernels and _importable("juliacall"):
            return True, "Julia kernel sources and juliacall are present."
        missing = [
            what
            for what, found in (
                ("Julia kernel sources", kernels),
                ("juliacall", _importable("juliacall")),
            )
            if not found
        ]
        return False, f"Missing here: {', '.join(missing)}."
    library = "Go" if lane == "go" else "Mojo"
    if _has_files(accel_root / lane, "*.so"):
        return True, f"Built {library} shared libraries are present."
    return False, f"No built {library} shared libraries are present."


def lane_statuses(*, accel_root: Path = ACCEL_ROOT) -> tuple[LaneStatus, ...]:
    """Report, for every lane, whether this installation has its resources.

    Parameters
    ----------
    accel_root:
        The ``sc_neurocore.accel`` directory to inspect; this installation's by
        default.

    Returns
    -------
    tuple of LaneStatus
        One status per lane, in contract order. A lane without its resources
        carries both what is missing and the contract's requirement.
    """
    statuses: list[LaneStatus] = []
    for contract in CONTRACT:
        present, found = _present(contract.lane, accel_root)
        detail = found if present else f"{found} {contract.requirement}"
        statuses.append(LaneStatus(contract.lane, contract.distribution, present, detail))
    return tuple(statuses)


__all__ = [
    "ACCEL_ROOT",
    "CONTRACT",
    "Distribution",
    "Lane",
    "LaneContract",
    "LaneStatus",
    "lane_statuses",
]

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Typed JuliaCall replay boundary

"""Typed Julia namespace fields and callable values at the managed import boundary."""

from collections.abc import Callable, Mapping
from typing import Protocol


class JuliaNativeModule(Protocol):
    """Julia function values accepting scalar C addresses and ABI metadata."""

    sc_if_abi_version: Callable[[], int]
    replay_pointer: Callable[[int, int], int]
    buffer_pointer: Callable[[int, int, int, int], int]
    free_pointer: Callable[[int], None]


class QCFSJuliaModule(Protocol):
    """Julia QCFS function values accepting integer addresses and element counts."""

    sc_qcfs_abi_version: Callable[[], int]
    sc_qcfs_forward: Callable[[int, float, int, int, int], int]
    sc_qcfs_backward: Callable[[int, float, int, int, int, int, int], int]


class JuliaNamespace(Protocol):
    """A Julia namespace containing a maintained rooted module."""

    AnnToSnnNative: JuliaNativeModule
    QcfsAccel: QCFSJuliaModule


class JuliaThreads(Protocol):
    """Active Julia default-pool worker count, fixed during runtime initialization."""

    nthreads: Callable[[], int]


class JuliaOptions(Protocol):
    """Active Julia signal handling selected when the runtime initialized."""

    handle_signals: int


class JuliaBase(Protocol):
    """The Julia include function value for loading source into a rooted namespace."""

    include: Callable[[JuliaNamespace, str], object]
    Threads: JuliaThreads
    JLOptions: Callable[[], JuliaOptions]


class JuliaMain(Protocol):
    """The managed main namespace used solely for source loading."""

    Base: JuliaBase


class JuliaRuntime(Protocol):
    """Configured runtime fields; import must not resolve or install dependencies."""

    CONFIG: Mapping[str, object]
    Main: JuliaMain
    newmodule: Callable[[str], JuliaNamespace]

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — the one loader of Mojo shared libraries, with the instruction-set guard

"""Load a Mojo shared library only on a host whose instruction set can run it.

Every Mojo library is built for :data:`~sc_neurocore.accel.mojo.isa_baseline.MOJO_TARGET_CPU`. On a host below that
x86-64 psABI level, loading or calling one may execute an instruction the CPU lacks, which ends the whole process with
``SIGILL``; Python cannot catch it. :func:`load_mojo_library` therefore reads the host's level once
(:func:`sc_neurocore.accel.host_isa.host_level`) and, when it is below the build level or unknown, raises
:class:`MojoIsaRefusal` *before* calling :class:`ctypes.CDLL`. The refusal is an :class:`OSError`, so every loader
that already treats a missing library as an unavailable lane refuses the lane the same way, and the dispatch falls
through to the next lane and finally to the Python floor.
"""

from __future__ import annotations

import ctypes
import functools
import os

from sc_neurocore.accel.host_isa import at_least, host_level
from sc_neurocore.accel.mojo.isa_baseline import MOJO_TARGET_CPU


class MojoIsaRefusal(OSError):
    """A Mojo library was not loaded, because the host's instruction-set level is below the library's.

    Parameters
    ----------
    library : str
        The library path that was not loaded.
    host : str
        The host's psABI level, or ``"unknown"``.
    required : str
        The level the library is built for.
    """

    def __init__(self, library: str, host: str, required: str) -> None:
        self.library = library
        self.host = host
        self.required = required
        super().__init__(
            f"{library} is built for {required}; this host is {host}, so the Mojo lane is refused"
        )


@functools.cache
def _host() -> str:
    """Return the running host's psABI level, read once per process."""
    return host_level()


def refusal_for_host() -> MojoIsaRefusal | None:
    """Return the refusal every Mojo library meets on this host, or ``None`` when Mojo libraries may load.

    Returns
    -------
    MojoIsaRefusal or None
        The reason a requested Mojo lane is unavailable on this host, for diagnostics; ``None`` on a host at or
        above the build level.
    """
    host = _host()
    if at_least(host, MOJO_TARGET_CPU):
        return None
    return MojoIsaRefusal("every Mojo library", host, MOJO_TARGET_CPU)


def load_mojo_library(path: str | os.PathLike[str]) -> ctypes.CDLL:
    """Load the Mojo shared library at ``path`` when this host can run it.

    Parameters
    ----------
    path : str or path-like
        The library file.

    Returns
    -------
    ctypes.CDLL
        The loaded library.

    Raises
    ------
    MojoIsaRefusal
        When the host's level is below the build level or unknown. :class:`ctypes.CDLL` is not called.
    OSError
        When the host may load the library but the loader cannot, as :class:`ctypes.CDLL` raises it.
    """
    host = _host()
    if not at_least(host, MOJO_TARGET_CPU):
        raise MojoIsaRefusal(os.fspath(path), host, MOJO_TARGET_CPU)
    return ctypes.CDLL(os.fspath(path))


__all__ = ["MojoIsaRefusal", "load_mojo_library", "refusal_for_host"]

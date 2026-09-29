# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native QCFS array libraries and managed Julia module

"""Load configured QCFS array implementations behind one address-based interface.

Rust, Go and Mojo libraries export the C functions ``sc_qcfs_abi_version``,
``sc_qcfs_forward`` and ``sc_qcfs_backward``; the Julia module exports the same
functions inside the already configured JuliaCall runtime. Every provider
validates before writing and returns 0 or -1.
"""

import atexit
import ctypes
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import cast

from .if_julia import admitted_julia_runtime
from .if_julia_configuration import julia_configuration
from .if_julia_types import QCFSJuliaModule

ForwardCall = Callable[[int, float, int, int, int], int]
BackwardCall = Callable[[int, float, int, int, int, int, int], int]


@dataclass(frozen=True)
class QCFSNativeAPI:
    """A loaded provider root and its forward and backward array entry points.

    Parameters
    ----------
    library : object
        Loaded library or managed Julia interface kept alive by this record.
    forward : callable
        ``(steps, theta, x, count, output) -> status`` over integer addresses.
    backward : callable
        ``(steps, theta, x, upstream, count, input_gradient, threshold_gradient)
        -> status`` over integer addresses.
    """

    library: object
    forward: ForwardCall
    backward: BackwardCall


@lru_cache(maxsize=16)
def load_qcfs_library(path: str) -> QCFSNativeAPI:
    """Load a configured C library and require QCFS array ABI version one.

    Parameters
    ----------
    path : str
        Explicit owner-configured shared library; nothing is built or downloaded.

    Returns
    -------
    QCFSNativeAPI
        Typed forward and backward entry points of the loaded library.

    Raises
    ------
    RuntimeError
        Library unavailable, entry points absent or ABI version incompatible.
    """
    try:
        library = ctypes.CDLL(str(Path(path).expanduser().resolve()))
        library.sc_qcfs_abi_version.argtypes = []
        library.sc_qcfs_abi_version.restype = ctypes.c_uint32
        if library.sc_qcfs_abi_version() != 1:
            raise RuntimeError("native QCFS library has an incompatible array ABI")
        library.sc_qcfs_forward.argtypes = [
            ctypes.c_uint32,
            ctypes.c_double,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        library.sc_qcfs_forward.restype = ctypes.c_int32
        library.sc_qcfs_backward.argtypes = [
            ctypes.c_uint32,
            ctypes.c_double,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        library.sc_qcfs_backward.restype = ctypes.c_int32
    except (OSError, AttributeError) as error:
        raise RuntimeError("native QCFS library or array entry points unavailable") from error
    return QCFSNativeAPI(
        library,
        cast(ForwardCall, library.sc_qcfs_forward),
        cast(BackwardCall, library.sc_qcfs_backward),
    )


class QCFSJulia:
    """Keep the Julia QCFS module rooted and refuse calls once shutdown begins."""

    def __init__(self, module: QCFSJuliaModule) -> None:
        """Retain the module and its bound functions; stop calls once exit is pending."""
        self.module = module
        self._forward = module.sc_qcfs_forward
        self._backward = module.sc_qcfs_backward
        self.closed = False
        atexit.register(self._stop)

    def _stop(self) -> None:
        """Stop managed calls before JuliaCall shuts its runtime down."""
        self.closed = True

    def forward(self, steps: int, theta: float, x: int, count: int, output: int) -> int:
        """Quantise through the managed runtime; see ``sc_qcfs_forward``.

        Parameters
        ----------
        steps, theta : int, float
            Admitted grid and threshold.
        x, count, output : int
            Input address, element count and output address.

        Returns
        -------
        int
            Zero or -1 with the output unchanged.

        Raises
        ------
        RuntimeError
            Runtime shutdown has begun.
        """
        if self.closed:
            raise RuntimeError("Julia QCFS runtime is closing")
        return int(self._forward(steps, theta, x, count, output))

    def backward(
        self,
        steps: int,
        theta: float,
        x: int,
        upstream: int,
        count: int,
        inputs: int,
        thresholds: int,
    ) -> int:
        """Differentiate through the managed runtime; see ``sc_qcfs_backward``.

        Parameters
        ----------
        steps, theta : int, float
            Admitted grid and threshold.
        x, upstream, count : int
            Input and upstream addresses and the element count.
        inputs, thresholds : int
            Non-overlapping derivative output addresses.

        Returns
        -------
        int
            Zero or -1 with both outputs unchanged.

        Raises
        ------
        RuntimeError
            Runtime shutdown has begun.
        """
        if self.closed:
            raise RuntimeError("Julia QCFS runtime is closing")
        return int(self._backward(steps, theta, x, upstream, count, inputs, thresholds))


@lru_cache(maxsize=1)
def _load_julia(executable: str, project: str) -> QCFSNativeAPI:
    """Include the packaged QCFS module into the exact configured runtime."""
    if "torch" in sys.modules and "juliacall" not in sys.modules:
        raise RuntimeError("Julia QCFS must be initialized before importing PyTorch")
    try:
        runtime = admitted_julia_runtime(executable, project, "Julia QCFS")
        namespace = runtime.newmodule("SCNeuroCoreQCFS")
        runtime.Main.Base.include(
            namespace, str(Path(__file__).parent.parent / "accel/julia/conversion/qcfs.jl")
        )
        module = namespace.QcfsAccel
        if int(module.sc_qcfs_abi_version()) != 1:
            raise RuntimeError("Julia QCFS module has an incompatible array ABI")
    except Exception as error:
        raise RuntimeError("Julia QCFS managed runtime unavailable or incompatible") from error
    interface = QCFSJulia(module)
    return QCFSNativeAPI(interface, interface.forward, interface.backward)


def load_qcfs_julia() -> QCFSNativeAPI:
    """Require explicit offline Julia configuration and return the QCFS interface.

    Returns
    -------
    QCFSNativeAPI
        Managed Julia calls sharing the QCFS array contract.

    Raises
    ------
    RuntimeError
        Runtime, project, versions or configuration unavailable or incompatible,
        or the runtime is closing.
    """
    api = _load_julia(*julia_configuration("Julia QCFS"))
    if cast(QCFSJulia, api.library).closed:
        raise RuntimeError("Julia QCFS runtime is closing")
    return api


__all__ = ["QCFSJulia", "QCFSNativeAPI", "load_qcfs_julia", "load_qcfs_library"]

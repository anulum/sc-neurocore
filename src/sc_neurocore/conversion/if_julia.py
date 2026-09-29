# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Managed Julia owned dense IF adapter

"""Load an explicitly configured JuliaCall runtime and share the owned-buffer ABI."""

import atexit
import ctypes
import importlib
import sys
from functools import lru_cache
from pathlib import Path
from typing import cast

from .if_julia_configuration import julia_configuration
from .if_julia_types import JuliaNativeModule, JuliaRuntime
from .if_native_types import NativeAPI


def _address(value: object) -> int:
    """Extract the ABI address from the ctypes argument retained by the calling adapter."""
    return ctypes.cast(cast(ctypes.c_void_p, value), ctypes.c_void_p).value or 0


class JuliaInterface:
    """Keep the configured Julia module rooted while managing pointer calls and shutdown."""

    def __init__(self, runtime: JuliaRuntime, module: JuliaNativeModule) -> None:
        """Retain the supplying managed runtime and forbid calls once its exit hook is pending."""
        self.runtime = runtime
        self.module = module
        self.closed = False
        atexit.register(self._stop)

    def _stop(self) -> None:
        """Stop managed calls before JuliaCall shuts its runtime down; process teardown reclaims remaining owners."""
        self.closed = True

    def replay(self, request: object, result: object) -> int:
        """Call the complete native request through JuliaCall's registered-thread entry.

        Parameters
        ----------
        request, result : ctypes pointer arguments
            Borrowed live request and exclusive writable result metadata.

        Returns
        -------
        int
            Native admission/overflow status; refusal does not write the owner slot.

        Raises
        ------
        RuntimeError
            Runtime shutdown has begun.
        """
        if self.closed:
            raise RuntimeError("Julia IF runtime is closing")
        return int(self.module.replay_pointer(_address(request), _address(result)))

    def buffer(self, handle: object, kind: int, index: int, view: object) -> int:
        """Borrow a rooted numeric vector while keeping managed runtime entry safe.

        Parameters
        ----------
        handle, view : ctypes pointer arguments
            Live owner and exclusive writable result view.
        kind, index : int
            ABI buffer selector and zero-based layer index.

        Returns
        -------
        int
            Zero success or negative native refusal without modifying view.

        Raises
        ------
        RuntimeError
            Runtime shutdown has begun.
        """
        if self.closed:
            raise RuntimeError("Julia IF runtime is closing")
        return int(
            self.module.buffer_pointer(
                _address(handle),
                ctypes.c_uint32(kind).value,
                ctypes.c_size_t(index).value,
                _address(view),
            )
        )

    def free(self, handle: object) -> None:
        """Release an expired view owner's rooted storage through the managed runtime.

        Parameters
        ----------
        handle : ctypes pointer argument
            Live supplying owner, released exactly once; null is harmless.
        """
        if not self.closed:
            self.module.free_pointer(_address(handle))


def admitted_julia_runtime(executable: str, project: str, label: str) -> JuliaRuntime:
    """Return the running JuliaCall runtime only if it is the exact configured one.

    Parameters
    ----------
    executable, project : str
        Resolved configured Julia executable and locked project.
    label : str
        Runtime user named in refusals, such as ``"Julia IF"``.

    Returns
    -------
    JuliaRuntime
        The imported JuliaCall module with one thread and Julia signal handling.

    Raises
    ------
    RuntimeError
        JuliaCall started with another executable, project, thread count or
        signal setting.
    ImportError
        JuliaCall is not installed.
    """
    runtime = cast(JuliaRuntime, importlib.import_module("juliacall"))
    if Path(cast(str, runtime.CONFIG["exepath"])).resolve() != Path(executable) or Path(
        cast(str, runtime.CONFIG["project"])
    ).resolve() != Path(project):
        raise RuntimeError(f"{label} runtime configuration changed after startup")
    if (
        runtime.Main.Base.Threads.nthreads() != 1
        or runtime.Main.Base.JLOptions().handle_signals != 1
    ):
        raise RuntimeError(f"{label} initialized thread or signal settings are incompatible")
    return runtime


@lru_cache(maxsize=1)
def _load(executable: str, project: str) -> NativeAPI:
    """Load the packaged boundary only into the exact already configured JuliaCall runtime."""
    if "torch" in sys.modules and "juliacall" not in sys.modules:
        raise RuntimeError("Julia IF must be initialized before importing PyTorch")
    try:
        runtime = admitted_julia_runtime(executable, project, "Julia IF")
        namespace = runtime.newmodule("SCNeuroCoreOwnedIF")
        runtime.Main.Base.include(
            namespace,
            str(Path(__file__).parent.parent / "accel/julia/conversion/ann_to_snn_native.jl"),
        )
        module = namespace.AnnToSnnNative
        if int(module.sc_if_abi_version()) != 1:
            raise RuntimeError("Julia IF library has incompatible ownership ABI")
    except Exception as error:
        raise RuntimeError("Julia IF managed runtime unavailable or incompatible") from error
    interface = JuliaInterface(runtime, module)
    return NativeAPI(interface, interface.replay, interface.buffer, interface.free)


def load_julia() -> NativeAPI:
    """Require explicit offline Julia configuration and obtain its owned replay interface.

    Returns
    -------
    NativeAPI
        Managed Julia calls sharing the complete ABI-one request/view contract.

    Raises
    ------
    RuntimeError
        Runtime/project/versions/configuration unavailable or incompatible.
    """
    api = _load(*julia_configuration())
    if cast(JuliaInterface, api.library).closed:
        raise RuntimeError("Julia IF runtime is closing")
    return api

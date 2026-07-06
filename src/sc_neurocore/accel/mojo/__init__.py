# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — High-level Mojo SIMD kernel API bounds

"""Mojo acceleration namespace.

Only a narrow subset of this tree is an authoritative backend surface today:

- Python loaders such as ``runner.py``
- compiled shared-library paths explicitly consumed from maintained Python code

Large parts of ``kernels/*.mojo`` are exploratory mirrors or transcripts and
must not be treated as the source of truth for runtime behaviour.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

AUTHORITATIVE_MOJO_ENTRYPOINTS: tuple[str, ...] = (
    "fault_injection/fault.mojo",
    "runner.py",
    "wilson_cowan/__init__.py",
    "wong_wang/__init__.py",
    "world_model/lgssm.mojo",
)

NON_AUTHORITATIVE_MOJO_MIRROR_GLOBS: tuple[str, ...] = (
    "kernels/*.mojo",
    "kernels/app.mojo",
    "kernels/compiler.mojo",
)

_HAS_MOJO = False
_mojo_import_reason: str | None = None
MOJO_HELPER_BACKEND: str = "unavailable"
MOJO_HELPER_IPC_AVAILABLE: bool = False

if TYPE_CHECKING:
    from .runner import MojoKernelRunner
else:

    class _UnavailableMojoKernelRunner:
        """Fail-closed placeholder for a missing optional Mojo runner."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            """Raise with the captured import failure instead of constructing."""
            reason = _mojo_import_reason or "unknown import failure"
            raise RuntimeError(f"Mojo runner unavailable: {reason}")

    try:
        from .runner import (
            MOJO_HELPER_BACKEND as _LOADED_MOJO_HELPER_BACKEND,
            MOJO_HELPER_IPC_AVAILABLE as _LOADED_MOJO_HELPER_IPC_AVAILABLE,
            MojoKernelRunner as _LoadedMojoKernelRunner,
        )

        MojoKernelRunner = _LoadedMojoKernelRunner
        MOJO_HELPER_BACKEND = _LOADED_MOJO_HELPER_BACKEND
        MOJO_HELPER_IPC_AVAILABLE = _LOADED_MOJO_HELPER_IPC_AVAILABLE
        _HAS_MOJO = True
    except Exception as _mojo_import_error:  # noqa: BLE001
        # Mojo toolchain missing or runner refused to import — record why so
        # downstream callers can log + surface the reason instead of silently
        # falling back.
        _HAS_MOJO = False
        _mojo_import_reason = repr(_mojo_import_error)
        MojoKernelRunner = _UnavailableMojoKernelRunner

__all__ = [
    "AUTHORITATIVE_MOJO_ENTRYPOINTS",
    "MOJO_HELPER_BACKEND",
    "MOJO_HELPER_IPC_AVAILABLE",
    "MojoKernelRunner",
    "NON_AUTHORITATIVE_MOJO_MIRROR_GLOBS",
    "_HAS_MOJO",
]

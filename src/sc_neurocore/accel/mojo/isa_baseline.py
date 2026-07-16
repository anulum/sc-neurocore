# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — portable ISA baseline for every Mojo build/run invocation
"""Central Mojo ISA baseline pin.

``mojo build`` / ``mojo run`` without an explicit ``--target-cpu`` compile for a
CPU baseline that may emit AVX-512. GitHub's hosted runners are heterogeneous:
a kernel compiled (or JIT-run) with AVX-512 on one runner raises SIGILL (``Fatal
Python error: Illegal instruction``) when the produced object is executed or
``ctypes``-loaded on a runner without AVX-512F. That surfaces as a
worker-microarchitecture lottery — green on one leg, red on another.

``x86-64-v3`` (AVX2 / FMA / BMI, no AVX-512) is present on every hosted runner,
so pinning it makes Mojo output run everywhere with identical numerics (a codegen
target, not a semantic change). Every Mojo ``build``/``run`` subprocess in this
repository must route its argv through :func:`pin_isa` so no single call site can
drift off the baseline; ``tools/check_mojo_isa_pin.py`` enforces this statically.
"""

from __future__ import annotations

from pathlib import Path

#: The portable baseline shared with the CI build step (.github/workflows/ci.yml)
#: and ``tools/build_accel_backends.py``. AVX2/FMA/BMI only — never AVX-512.
MOJO_TARGET_CPU = "x86-64-v3"


def pin_isa(argv: list[str]) -> list[str]:
    """Return ``argv`` with ``--target-cpu x86-64-v3`` after its mojo subcommand.

    Finds the ``mojo`` executable followed by ``build`` or ``run`` and inserts
    the baseline flag immediately after the subcommand. The executable is matched
    by basename, so a resolved absolute path (e.g. ``shutil.which("mojo")`` →
    ``/home/user/.local/bin/mojo``) is pinned exactly like the bare ``mojo``
    token — otherwise an absolute-path call site would build an unpinned kernel
    that SIGILLs on a non-AVX-512 runner. A ``*.mojo`` source file is not matched
    (its basename ends in ``.mojo``, not ``mojo``). Idempotent: an argv that
    already carries ``--target-cpu`` is returned unchanged. A copy is returned;
    the input list is not mutated.
    """
    out = list(argv)
    if "--target-cpu" in out:
        return out
    for index, argument in enumerate(out):
        if (
            Path(argument).name == "mojo"
            and index + 1 < len(out)
            and out[index + 1] in ("build", "run")
        ):
            out[index + 2 : index + 2] = ["--target-cpu", MOJO_TARGET_CPU]
            return out
    return out

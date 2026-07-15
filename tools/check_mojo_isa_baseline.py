# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo accelerator ISA-baseline drift gate
"""Fail if any built Mojo accelerator ``.so`` uses instructions above the pinned baseline.

``tools/build_accel_backends.py`` pins ``mojo build --target-cpu x86-64-v3`` so the compiled
kernels stay within AVX2/FMA/BMI and run on any hosted CI runner. Nothing stops a new build
path (a test, a per-model recipe, a dropped flag) from silently re-emitting AVX-512, which
raises ``SIGILL`` on runners without AVX-512F. This gate objdump-scans every Mojo backend
library and rejects AVX-512 register use so the pin cannot regress unnoticed.

Only the Mojo libraries are scanned: the Go runtime carries AVX-512 code paths that are
guarded by runtime CPU-feature dispatch (``internal/cpu``) and never execute without support,
whereas Mojo emits the wide instructions unconditionally into the kernel.

    python tools/check_mojo_isa_baseline.py            # scan accel/mojo/**/lib*.so
    python tools/check_mojo_isa_baseline.py --root DIR # scan an explicit tree
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from collections.abc import Sequence

# AVX-512 markers in AT&T-syntax ``objdump -d`` output: 512-bit ``zmm`` data registers and the
# ``k0``–``k7`` opmask registers. Both are exclusive to AVX-512; AVX2 (x86-64-v3) uses ``ymm``.
_AVX512_MARKER = re.compile(r"%zmm[0-9]+|%k[0-7]\b")

_DEFAULT_ROOT = Path("src/sc_neurocore/accel/mojo")


def scan_disassembly(text: str) -> list[str]:
    """Return the AVX-512 instruction lines (zmm / opmask) present in ``objdump -d`` output."""
    return [line.strip() for line in text.splitlines() if _AVX512_MARKER.search(line)]


def scan_library(library: Path) -> list[str]:
    """Disassemble one shared object and return its AVX-512 instruction lines.

    A library ``objdump`` cannot read (missing tool, non-ELF, truncated build) is skipped with
    a warning rather than crashing the gate; a genuine backend build is always a valid ELF.
    """
    try:
        result = subprocess.run(
            ["objdump", "-d", str(library)],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"check_mojo_isa_baseline: could not disassemble {library}: {exc}", file=sys.stderr)
        return []
    return scan_disassembly(result.stdout)


def discover_libraries(root: Path) -> list[Path]:
    """Return our compiled Mojo backend ``lib*.so`` under ``root`` in stable order.

    Excludes the ``.pixi`` environment: those are the pixi-installed Mojo/LLVM/LLDB toolchain
    libraries (hundreds of MB, and legitimately full of AVX-512), not the small kernels we emit.
    """
    return sorted(p for p in root.rglob("lib*.so") if ".pixi" not in p.parts)


def check(root: Path) -> dict[Path, list[str]]:
    """Map each offending library to its AVX-512 instruction lines (empty map == clean)."""
    offenders: dict[Path, list[str]] = {}
    for library in discover_libraries(root):
        hits = scan_library(library)
        if hits:
            offenders[library] = hits
    return offenders


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reject AVX-512 in Mojo accelerator libraries.")
    parser.add_argument("--root", type=Path, default=_DEFAULT_ROOT, help="Tree of Mojo lib*.so.")
    args = parser.parse_args(argv)

    if not args.root.is_dir():
        print(f"check_mojo_isa_baseline: no Mojo backend tree at {args.root}", file=sys.stderr)
        return 0

    libraries = discover_libraries(args.root)
    if not libraries:
        print(f"check_mojo_isa_baseline: no lib*.so under {args.root} (nothing built yet)")
        return 0

    offenders = check(args.root)
    if not offenders:
        print(
            f"check_mojo_isa_baseline: {len(libraries)} Mojo librar(ies) within x86-64-v3 baseline"
        )
        return 0

    for library, hits in offenders.items():
        print(
            f"ISA DRIFT: {library} carries AVX-512 above the x86-64-v3 baseline:", file=sys.stderr
        )
        for line in hits[:5]:
            print(f"    {line}", file=sys.stderr)
    print(
        "Pin --target-cpu x86-64-v3 on every Mojo build path "
        "(see tools/build_accel_backends.py::_mojo_command).",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

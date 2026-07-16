# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Third-party wheel ISA report-only scanner
"""Report third-party native wheels that carry unguarded AVX-512 (a SIGILL risk).

``tools/check_mojo_isa_baseline.py`` gates *our* Mojo kernels. This sibling scans the
*installed third-party* native ``.so`` in a CI environment (numpy, gdstk, klayout, orjson…)
and flags any library that emits AVX-512 register use **without** a runtime CPU-feature
check. GitHub-hosted ``ubuntu-latest`` runners are a heterogeneous CPU fleet: a wheel built
with a fixed AVX-512 target (no ``cpuid`` dispatch) links and imports fine on an AVX-512
build host and on many runners, then raises ``SIGILL`` (exit 132) the moment it lands on a
runner whose CPU lacks AVX-512F — an intermittent "runner lottery" red that leaves no test
output. This scanner surfaces the risky wheel **at pin time** instead of in a red matrix.

The signal is a heuristic, so this is report-only by default (exit 0):

* AVX-512 registers present **and** no ``cpuid`` in the object  → ``unguarded`` (SIGILL risk):
  the wide instructions cannot be gated by runtime detection the object never performs.
* AVX-512 registers present **and** ``cpuid`` present            → ``dispatched`` (informational):
  the object detects CPU features at runtime and is expected to pick a safe kernel.
* no AVX-512 registers                                            → clean (not reported).

``cpuid`` presence does not *prove* the specific AVX-512 block is guarded, so ``dispatched``
is advisory; ``unguarded`` is the actionable finding. Pass ``--strict`` to exit non-zero when
any ``unguarded`` library is found (for use as a pinning gate).

    python tools/scan_wheel_isa.py --root .venv/lib/python3.12/site-packages
    python tools/scan_wheel_isa.py --root DIR --strict
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# AVX-512 markers in AT&T-syntax ``objdump -d`` output: 512-bit ``zmm`` data registers and the
# ``k0``–``k7`` opmask registers, both exclusive to AVX-512. Matches the marker used by the
# sibling gate ``tools/check_mojo_isa_baseline.py`` so the two ISA tools agree on the signal.
_AVX512_MARKER = re.compile(r"%zmm[0-9]+|%k[0-7]\b")

# A ``cpuid`` instruction is how a library performs runtime CPU-feature detection before
# dispatching to a wide-SIMD kernel; its absence beside AVX-512 use means the object cannot
# be gating that use on the host actually supporting it.
_CPUID_MARKER = re.compile(r"\bcpuid\b")


@dataclass(frozen=True)
class LibraryScan:
    """AVX-512 / cpuid evidence for one shared object."""

    library: Path
    avx512_hits: int
    cpuid_hits: int

    @property
    def risk(self) -> str:
        """``unguarded`` (AVX-512, no cpuid), ``dispatched`` (AVX-512 + cpuid) or ``clean``."""
        if self.avx512_hits == 0:
            return "clean"
        return "dispatched" if self.cpuid_hits > 0 else "unguarded"


def scan_disassembly(text: str) -> tuple[int, int]:
    """Return ``(avx512_line_count, cpuid_instruction_count)`` for ``objdump -d`` output."""
    avx512 = 0
    cpuid = 0
    for line in text.splitlines():
        if _AVX512_MARKER.search(line):
            avx512 += 1
        if _CPUID_MARKER.search(line):
            cpuid += 1
    return avx512, cpuid


def scan_library(library: Path) -> LibraryScan:
    """Disassemble one shared object and return its AVX-512 / cpuid evidence.

    A library ``objdump`` cannot read (missing tool, non-ELF, truncated) is reported as clean
    with a warning rather than crashing the scan; a report-only tool must never abort a build.
    """
    try:
        result = subprocess.run(
            ["objdump", "-d", str(library)],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"scan_wheel_isa: could not disassemble {library}: {exc}", file=sys.stderr)
        return LibraryScan(library=library, avx512_hits=0, cpuid_hits=0)
    avx512, cpuid = scan_disassembly(result.stdout)
    return LibraryScan(library=library, avx512_hits=avx512, cpuid_hits=cpuid)


def discover_libraries(root: Path) -> list[Path]:
    """Return every native ``*.so`` under ``root`` in stable order.

    Third-party extensions are named for their module (``orjson.cpython-…so``,
    ``_multiarray_umath.…so``) and their bundled shared libraries live in ``*.libs`` siblings,
    so the whole tree is walked rather than matching a ``lib`` prefix.
    """
    return sorted(root.rglob("*.so"))


def scan(root: Path) -> list[LibraryScan]:
    """Scan every native library under ``root`` and return only the AVX-512-carrying ones."""
    findings = [scan_library(library) for library in discover_libraries(root)]
    return [finding for finding in findings if finding.avx512_hits > 0]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report third-party native wheels carrying unguarded AVX-512 (SIGILL risk)."
    )
    parser.add_argument(
        "--root", type=Path, required=True, help="Directory to scan (e.g. a venv site-packages)."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when any unguarded AVX-512 library is found.",
    )
    args = parser.parse_args(argv)

    if not args.root.is_dir():
        print(f"scan_wheel_isa: no directory at {args.root}", file=sys.stderr)
        return 0

    findings = scan(args.root)
    unguarded = [f for f in findings if f.risk == "unguarded"]
    dispatched = [f for f in findings if f.risk == "dispatched"]

    for finding in sorted(dispatched, key=lambda f: f.library.as_posix()):
        rel = finding.library.relative_to(args.root)
        print(f"dispatched  {rel}  (avx512={finding.avx512_hits}, cpuid={finding.cpuid_hits})")
    for finding in sorted(unguarded, key=lambda f: f.library.as_posix()):
        rel = finding.library.relative_to(args.root)
        print(
            f"UNGUARDED   {rel}  (avx512={finding.avx512_hits}, cpuid=0) — SIGILL risk on "
            "runners without AVX-512"
        )

    print(
        f"scan_wheel_isa: {len(findings)} librar(ies) with AVX-512 "
        f"({len(unguarded)} unguarded, {len(dispatched)} dispatched)"
    )
    if unguarded and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

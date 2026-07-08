#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Criterion benchmark converter
"""Convert Criterion 0.5 stdout to bencher format for benchmark-action."""

import re
import sys
from collections.abc import Iterator

# Nanoseconds per Criterion time unit.
_SCALE = {"ns": 1, "µs": 1_000, "ms": 1_000_000, "s": 1_000_000_000}
# One "<value> <unit>" estimate. Longer units precede the bare "s" so "µs"/"ms"/"ns"
# never match as a bare second. This pairs each number with its OWN unit — Criterion
# formats the "[low mid high]" estimate triplet and, when a benchmark straddles a unit
# boundary, mixes units within it, e.g. "[999.50 µs 1.0001 ms 1.0050 ms]". Reading the
# median value with the first unit (as a naive split does) then scales 1.0001 as µs → 1000
# ns instead of 1000100 ns — a ~1000× under-read that poisons the benchmark baseline.
_ESTIMATE = re.compile(r"([\d.]+)\s*(µs|ms|ns|s)")


def convert(text: str) -> Iterator[str]:
    """Yield ``test <name> ... bench: <ns> ns/iter (+/- 0)`` lines from Criterion stdout.

    The middle estimate of the ``time: [low median high]`` triplet is emitted, converted
    to nanoseconds using the median's own unit.
    """
    name: str | None = None
    for line in text.splitlines():
        line = line.rstrip()
        # Name either shares the result line ("<name>  time: [...]") or is a standalone line.
        same_line = re.match(r"^(\S+)\s+time:", line)
        if same_line:
            name = same_line.group(1)
        elif re.match(r"^\S", line) and "time:" not in line and "change:" not in line:
            name = line.strip()

        if "time:" in line and name and "[" in line and "]" in line:
            bracket = line.split("[", 1)[1].split("]", 1)[0]
            pairs = _ESTIMATE.findall(bracket)
            if len(pairs) >= 2:
                median_value, median_unit = pairs[1]
                ns = float(median_value) * _SCALE[median_unit]
                yield f"test {name} ... bench: {int(ns)} ns/iter (+/- 0)"
            name = None


def main() -> None:
    """Read Criterion stdout and write bencher-format lines to stdout."""
    for out in convert(sys.stdin.read()):
        print(out)


if __name__ == "__main__":
    main()

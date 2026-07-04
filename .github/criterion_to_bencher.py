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

name = None
for line in sys.stdin:
    line = line.rstrip()
    # Name on same line as time, or standalone name line
    m = re.match(r"^(\S+)\s+time:", line)
    if m:
        name = m.group(1)
    elif re.match(r"^\S", line) and "time:" not in line and "change:" not in line:
        name = line.strip()

    # Extract median (second value in brackets) from time: lines
    if "time:" in line and name:
        m2 = re.search(r"\[\S+\s+\S+\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\]", line)
        if not m2:
            m2 = re.search(r"\[(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\]", line)
        # Parse: [low median high unit]
        vals = re.findall(r"[\d.]+", line.split("[")[1].split("]")[0])
        units = re.findall(r"[µnm]?s", line.split("[")[1].split("]")[0])
        if len(vals) >= 2 and units:
            median = float(vals[1])
            unit = units[0]
            if unit == "ns":
                ns = median
            elif unit == "µs":
                ns = median * 1000
            elif unit == "ms":
                ns = median * 1_000_000
            elif unit == "s":
                ns = median * 1_000_000_000
            else:
                ns = median
            print(f"test {name} ... bench: {int(ns)} ns/iter (+/- 0)")
        name = None

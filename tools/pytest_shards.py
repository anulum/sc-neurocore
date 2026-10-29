# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Duration-weighted pytest module scheduling

"""Partition discovered modules and retain measured JUnit durations for later CI runs."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import heapq
import json
import math
from pathlib import Path
import statistics
import sys
from typing import cast

# JUnit reports arrive as downloaded CI artifacts; entity declarations are refused, not expanded.
from defusedxml import ElementTree as ET


def read_timings(path: Path | None) -> dict[str, float]:
    """Read optional scheduling hints; refuse malformed or nonfinite measurements.

    Parameters
    ----------
    path : Path or None
        Previous duration manifest. A missing file represents a cold run.

    Returns
    -------
    dict of str to float
        Positive seconds keyed by repository-relative test module.
    """
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != 1:
        raise ValueError("Unsupported pytest timing manifest")
    durations = data.get("durations")
    if not isinstance(durations, dict):
        raise ValueError("Timing durations must be an object")
    result: dict[str, float] = {}
    for name, seconds in cast(dict[str, object], durations).items():
        if (
            not isinstance(seconds, (int, float))
            or isinstance(seconds, bool)
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            raise ValueError("Timing entries require module names and positive finite seconds")
        result[name] = float(seconds)
    return result


def partition(files: list[str], count: int, timings: dict[str, float]) -> list[list[str]]:
    """Assign every discovered module once, balancing longest measured modules first.

    Parameters
    ----------
    files : list of str
        Complete selected discovery list; timing data never controls membership.
    count : int
        Number of nonempty shards.
    timings : dict of str to float
        Historical durations. New modules receive the median observed duration.

    Returns
    -------
    list of list of str
        Deterministic sorted module lists, one per shard.
    """
    if not 1 <= count <= len(files) or len(set(files)) != len(files) or not all(files):
        raise ValueError("Shard count must fit the unique discovered module list")
    if not timings:
        return [files[index::count] for index in range(count)]
    default = statistics.median(timings.values())
    shards: list[list[str]] = [[] for _ in range(count)]
    loads = [(0.0, index) for index in range(count)]
    heapq.heapify(loads)
    for name in sorted(files, key=lambda name: (-timings.get(name, default), name)):
        load, index = heapq.heappop(loads)
        shards[index].append(name)
        heapq.heappush(loads, (load + timings.get(name, default), index))
    return [sorted(shard) for shard in shards]


def record_timings(reports: Path, previous: Path | None, output: Path) -> None:
    """Aggregate actual pytest JUnit case durations by live repository module.

    Parameters
    ----------
    reports : Path
        Directory containing downloaded batch XML reports from coverage shards.
    previous : Path or None
        Previous manifest; unexecuted live modules retain their last measurement.
    output : Path
        Destination for the next run's scheduling hints.
    """
    modules = {
        str(path.with_suffix("")).replace("/", "."): path.as_posix()
        for path in Path("tests").rglob("*.py")
        if path.name.startswith("test_") or path.name.endswith("_test.py")
    }
    live = set(modules.values())
    durations = {name: value for name, value in read_timings(previous).items() if name in live}
    measured: dict[str, float] = {}
    for report in sorted(reports.rglob("test-results-batch-*.xml")):
        for case in ET.parse(report).iter("testcase"):
            name = case.get("classname", "")
            while name and name not in modules:
                name = name.rpartition(".")[0]
            if not name:
                continue
            seconds = float(case.get("time", "0"))
            if not math.isfinite(seconds) or seconds < 0:
                raise ValueError("JUnit durations must be nonnegative and finite")
            path = modules[name]
            measured[path] = measured.get(path, 0.0) + seconds
    if not measured:
        raise ValueError("No live module durations found in JUnit reports")
    durations.update({name: max(seconds, 0.001) for name, seconds in measured.items()})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"schema": 1, "durations": durations}, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the scheduler CLI using NUL-separated discovery or actual JUnit reports.

    Parameters
    ----------
    argv : sequence of str or None
        Command arguments, defaulting to this process's arguments.

    Returns
    -------
    int
        Zero for a complete selection or measurement; two for invalid inputs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select")
    select.add_argument("--shard", required=True)
    select.add_argument("--timings", type=Path)
    record = commands.add_parser("record")
    record.add_argument("--reports", type=Path, required=True)
    record.add_argument("--previous", type=Path)
    record.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "record":
            record_timings(args.reports, args.previous, args.output)
        else:
            index, count = (int(value) for value in args.shard.split("/"))
            if not 1 <= index <= count:
                raise ValueError("Shard must be K/N with 1 <= K <= N")
            timings = read_timings(args.timings)
            data = sys.stdin.buffer.read()
            if not data.endswith(b"\0"):
                raise ValueError("Discovery must be NUL-terminated")
            files = data[:-1].decode().split("\0")
            selected = partition(files, count, timings)[index - 1]
            sys.stdout.buffer.write(b"\0".join(name.encode() for name in selected) + b"\0")
    except (OSError, ValueError, ET.ParseError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

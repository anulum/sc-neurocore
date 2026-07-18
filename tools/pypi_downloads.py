#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — daily PyPI download snapshot, recorded to a CSV time series
"""Upsert the available daily PyPI download history into a CSV time series."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

PYPISTATS_OVERALL = "https://pypistats.org/api/packages/{package}/overall"
CATEGORIES = ("without_mirrors", "with_mirrors")
RETRYABLE_STATUSES = (429, 502, 503, 504)
RETRY_SCHEDULE = (30.0, 60.0, 120.0)
MAX_RETRY_AFTER = 600.0

Fetch = Callable[[str], bytes]
Sleep = Callable[[float], None]


def detect_package(pyproject_path: Path) -> str:
    """Return the distribution name from ``pyproject.toml``."""
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    name = str(data.get("project", {}).get("name", "")).strip()
    if not name:
        raise ValueError(f"no [project] name in {pyproject_path}")
    return name


def _http_get(url: str) -> bytes:
    """Fetch the fixed HTTPS statistics endpoint and return its response body."""
    with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
        body: bytes = response.read()
        return body


def fetch_overall(package: str, fetch: Fetch = _http_get) -> dict[str, Any]:
    """Fetch and decode pypistats' overall payload for ``package``."""
    raw = fetch(PYPISTATS_OVERALL.format(package=package))
    decoded: dict[str, Any] = json.loads(raw)
    return decoded


def _retry_delay(exc: urllib.error.HTTPError, fallback: float) -> float:
    """Use a bounded Retry-After value, falling back to the local schedule."""
    header = str(exc.headers.get("Retry-After", "")) if exc.headers else ""
    try:
        delay = float(header)
    except ValueError:
        return fallback
    return min(max(delay, 1.0), MAX_RETRY_AFTER)


def fetch_overall_with_retry(
    package: str, fetch: Fetch = _http_get, sleep: Sleep = time.sleep
) -> dict[str, Any] | None:
    """Retry transient upstream errors and return ``None`` after exhaustion."""
    for delay in (*RETRY_SCHEDULE, None):
        try:
            return fetch_overall(package, fetch)
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRYABLE_STATUSES:
                raise
            if delay is not None:
                sleep(_retry_delay(exc, delay))
    return None


def daily_counts(overall: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Reduce the upstream payload to per-date counts for known categories."""
    counts: dict[str, dict[str, int]] = {}
    for row in overall.get("data", []):
        category = str(row.get("category", ""))
        date = str(row.get("date", "")).strip()
        if category not in CATEGORIES or not date:
            continue
        try:
            downloads = int(row.get("downloads", 0))
        except (TypeError, ValueError):
            continue
        counts.setdefault(date, {})[category] = downloads
    return counts


def read_csv(path: Path) -> dict[str, dict[str, int]]:
    """Read an existing snapshot CSV into a date-keyed mapping."""
    if not path.exists():
        return {}
    rows: dict[str, dict[str, int]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            date = (record.get("date") or "").strip()
            if date:
                rows[date] = {category: int(record.get(category) or 0) for category in CATEGORIES}
    return rows


def merge(
    existing: dict[str, dict[str, int]], fresh: dict[str, dict[str, int]]
) -> dict[str, dict[str, int]]:
    """Upsert fresh counts without dropping dates outside the upstream window."""
    merged = {date: dict(values) for date, values in existing.items()}
    for date, values in fresh.items():
        merged.setdefault(date, {}).update(values)
    return merged


def write_csv(path: Path, rows: dict[str, dict[str, int]]) -> None:
    """Write a stable, date-sorted CSV time series."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", *CATEGORIES])
        for date in sorted(rows):
            writer.writerow([date, *(rows[date].get(category, 0) for category in CATEGORIES)])


def _summary(package: str, rows: dict[str, dict[str, int]]) -> str:
    """Summarize the latest recorded date for the Actions log."""
    if not rows:
        return f"{package}: no download data available yet"
    latest = max(rows)
    values = rows[latest]
    return (
        f"{package}: {len(rows)} days recorded; latest {latest} "
        f"without_mirrors={values.get('without_mirrors', 0)} "
        f"with_mirrors={values.get('with_mirrors', 0)}"
    )


def main(argv: list[str] | None = None, fetch: Fetch = _http_get, sleep: Sleep = time.sleep) -> int:
    """Fetch, merge, and persist the package's download history."""
    parser = argparse.ArgumentParser(description="Record daily PyPI download history.")
    parser.add_argument("--pyproject", default="pyproject.toml")
    parser.add_argument("--package", default=None)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--print-package", action="store_true")
    args = parser.parse_args(argv)

    package = args.package or detect_package(Path(args.pyproject))
    if args.print_package:
        print(package)
        return 0
    if not args.csv:
        parser.error("--csv is required unless --print-package is given")

    try:
        overall = fetch_overall_with_retry(package, fetch, sleep)
    except urllib.error.URLError as exc:
        print(f"{package}: could not fetch download stats: {exc}", file=sys.stderr)
        return 1
    if overall is None:
        print(
            f"{package}: upstream rate limit persisted across retries; "
            "skipping this snapshot (the per-date upsert self-heals next run)",
            file=sys.stderr,
        )
        return 0

    csv_path = Path(args.csv)
    merged = merge(read_csv(csv_path), daily_counts(overall))
    write_csv(csv_path, merged)
    print(_summary(package, merged))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — analytic side-channel benchmark tool

"""Generate an analytic side-channel benchmark JSON report."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from sc_neurocore.security import (
    SideChannelBenchmarkError,
    ThermalSCEncodingConfig,
    write_side_channel_benchmark_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Output report JSON path")
    parser.add_argument(
        "--probabilities",
        required=True,
        help="Comma-separated sample probabilities in [0, 1]",
    )
    parser.add_argument("--labels", required=True, help="Comma-separated numeric labels")
    parser.add_argument("--bitstream-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rotation-stride", type=int, default=1)
    parser.add_argument("--dummy-streams-per-record", type=int, default=0)
    parser.add_argument("--max-dummy-overhead-ratio", type=float, default=0.0)
    args = parser.parse_args(argv)

    try:
        probabilities = _parse_float_csv(args.probabilities, "probabilities")
        labels = _parse_float_csv(args.labels, "labels")
        write_side_channel_benchmark_report(
            args.output,
            probabilities=probabilities,
            labels=labels,
            protected_config=ThermalSCEncodingConfig(
                bitstream_length=args.bitstream_length,
                seed=args.seed,
                rotation_stride=args.rotation_stride,
                dummy_streams_per_record=args.dummy_streams_per_record,
                max_dummy_overhead_ratio=args.max_dummy_overhead_ratio,
            ),
        )
    except (SideChannelBenchmarkError, ValueError, OSError) as exc:
        print(f"side-channel benchmark invalid: {exc}", file=sys.stderr)
        return 1
    print(args.output)
    return 0


def _parse_float_csv(raw: str, field_name: str) -> tuple[float, ...]:
    values: list[float] = []
    for part in raw.split(","):
        stripped = part.strip()
        if not stripped:
            raise ValueError(f"{field_name} contains an empty value")
        values.append(float(stripped))
    if not values:
        raise ValueError(f"{field_name} must not be empty")
    return tuple(values)


if __name__ == "__main__":
    raise SystemExit(main())

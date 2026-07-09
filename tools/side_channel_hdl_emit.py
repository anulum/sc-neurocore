#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — side-channel HDL hook emission tool

"""Emit a protected side-channel encoding HDL hook and manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from sc_neurocore.hdl_gen import SideChannelEncodingEmitter
from sc_neurocore.security import (
    ThermalSCEncodingConfig,
    ThermalSCEncodingError,
    encode_activity_balanced_probability,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verilog-output", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    parser.add_argument("--module-name", default="sc_side_channel_encoding_source")
    parser.add_argument("--probability", required=True, type=float)
    parser.add_argument("--bitstream-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rotation-stride", type=int, default=1)
    parser.add_argument("--dummy-streams-per-record", type=int, default=0)
    parser.add_argument("--max-dummy-overhead-ratio", type=float, default=0.0)
    args = parser.parse_args(argv)

    try:
        record = encode_activity_balanced_probability(
            args.probability,
            ThermalSCEncodingConfig(
                bitstream_length=args.bitstream_length,
                seed=args.seed,
                rotation_stride=args.rotation_stride,
                dummy_streams_per_record=args.dummy_streams_per_record,
                max_dummy_overhead_ratio=args.max_dummy_overhead_ratio,
            ),
        )
        emitter = SideChannelEncodingEmitter(
            module_name=args.module_name,
            encoding=record,
        )
        verilog = emitter.generate()
        manifest = emitter.manifest(verilog_path=str(args.verilog_output))
        args.verilog_output.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
        args.verilog_output.write_text(verilog + "\n", encoding="utf-8")
        args.manifest_output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (ThermalSCEncodingError, ValueError, OSError) as exc:
        print(f"side-channel HDL hook invalid: {exc}", file=sys.stderr)
        return 1

    print(args.verilog_output)
    print(args.manifest_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

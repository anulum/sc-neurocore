#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Stochastic backpropagation benchmark tool

"""Generate a deterministic stochastic backpropagation benchmark JSON report."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from sc_neurocore.benchmarks import build_stochastic_backprop_benchmark
from sc_neurocore.benchmarks.stochastic_backprop import (
    write_stochastic_backprop_estimator_regression_manifest,
)
from sc_neurocore.training.sc_estimators import DifferentiableSCConfig
from sc_neurocore.training.stochastic_backprop_export import (
    build_stochastic_backprop_export_manifest,
    write_stochastic_backprop_handoff_bundle,
    write_stochastic_backprop_export_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Output report JSON path")
    parser.add_argument("--export-manifest", type=Path, default=None)
    parser.add_argument("--estimator-regression-manifest", type=Path, default=None)
    parser.add_argument("--handoff-dir", type=Path, default=None)
    parser.add_argument("--bitstream-length", type=int, default=256)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.4)
    args = parser.parse_args(argv)

    try:
        payload = build_stochastic_backprop_benchmark(
            bitstream_length=args.bitstream_length,
            steps=args.steps,
            learning_rate=args.learning_rate,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        import json

        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if args.export_manifest is not None or args.handoff_dir is not None:
            sc_config = payload["sc_config"]
            config = DifferentiableSCConfig(
                bitstream_length=int(sc_config["bitstream_length"]),
                encoding=sc_config["encoding"],
                generator=sc_config["generator"],
                estimator=sc_config["estimator"],
                input_seed=int(sc_config["input_seed"]),
                weight_seed=int(sc_config["weight_seed"]),
                correlation=float(sc_config["correlation"]),
            )
        if args.export_manifest is not None:
            write_stochastic_backprop_export_manifest(
                args.export_manifest,
                payload,
                config,
            )
        if args.estimator_regression_manifest is not None:
            write_stochastic_backprop_estimator_regression_manifest(
                args.estimator_regression_manifest,
                bitstream_lengths=(64, 128, 256),
                sample_count=32,
            )
        if args.handoff_dir is not None:
            manifest = build_stochastic_backprop_export_manifest(payload, config)
            report = write_stochastic_backprop_handoff_bundle(manifest, args.handoff_dir)
            audit_path = args.handoff_dir / "stochastic_backprop_handoff_audit.json"
            audit_path.write_text(
                json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"stochastic backpropagation benchmark invalid: {exc}", file=sys.stderr)
        return 1
    print(args.output)
    if args.export_manifest is not None:
        print(args.export_manifest)
    if args.estimator_regression_manifest is not None:
        print(args.estimator_regression_manifest)
    if args.handoff_dir is not None:
        print(args.handoff_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

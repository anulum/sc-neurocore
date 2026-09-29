# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Five-runtime owned dense IF comparison

"""Run all five public runtimes sequentially in fresh, equally pinned processes."""

import argparse
from collections.abc import Sequence
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import cast

from sc_neurocore.conversion.if_dispatch import ReplayBackend
from _ann_to_snn_replay_measurement import (
    BACKENDS,
    KERNEL,
    REPOSITORY,
    configured_artifacts,
    measure,
    metadata,
    source_digests,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Capture a complete source/artifact-bound comparison through the real public API.

    Parameters
    ----------
    argv : sequence of str or None
        Explicit CLI arguments; None reads the actual command line.

    Returns
    -------
    int
        Zero after all five verified runtimes and atomic report publication.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--worker", choices=BACKENDS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.samples < 3 or args.warmup < 1:
        parser.error("at least three samples and one warmup required")
    if args.cpu not in os.sched_getaffinity(0):
        parser.error("CPU must belong to the current allowed affinity")
    os.sched_setaffinity(0, {args.cpu})
    if args.worker is not None:
        print(json.dumps(measure(cast(ReplayBackend, args.worker), args.samples, args.warmup)))
        return 0
    sources = source_digests()
    artifacts = configured_artifacts()
    context = metadata(args.cpu, args.samples, args.warmup)
    results: dict[str, object] = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for backend in BACKENDS:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--output",
            str(args.output),
            "--cpu",
            str(args.cpu),
            "--samples",
            str(args.samples),
            "--warmup",
            str(args.warmup),
            "--worker",
            backend,
        ]
        completed = subprocess.run(
            command,
            cwd=REPOSITORY,
            env=dict(os.environ, GOMAXPROCS="1"),
            capture_output=True,
            text=True,
            timeout=300,
        )
        args.output.with_suffix(f".{backend}.stdout").write_text(completed.stdout)
        args.output.with_suffix(f".{backend}.stderr").write_text(completed.stderr)
        if completed.returncode:
            raise RuntimeError(f"{backend} comparison failed; see retained stdout/stderr")
        results[backend] = json.loads(completed.stdout)
    if sources != source_digests() or artifacts != configured_artifacts():
        raise RuntimeError("runtime sources or configured artifacts changed during capture")
    context["load_average_after"] = list(os.getloadavg())
    payload = {
        "schema_version": "sc-neurocore.dense-if-comparison.v1",
        "kernel": KERNEL,
        "meta": context,
        "source_sha256": sources,
        "artifact_sha256": artifacts,
        "backends": results,
    }
    with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False) as temporary:
        json.dump(payload, temporary, indent=2)
        temporary.write("\n")
        temporary.flush()
        os.fsync(temporary.fileno())
    os.replace(temporary.name, args.output)
    print(f"All five runtimes verified; wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source-bound measured dense IF dispatch order

"""Use an explicitly supplied complete comparison only for its matching live runtime."""

import importlib.metadata
import json
import os
import platform
from pathlib import Path

from .if_benchmark_identity import configured_artifacts, source_digests
from .if_benchmark_record import NATIVE_BACKENDS, validated_timing_order

STATIC_ORDER = NATIVE_BACKENDS


def measured_order() -> tuple[str, ...]:
    """Resolve host-matched measured native ordering without importing any native runtime.

    Returns
    -------
    tuple of str
        Native providers ordered by full-response warm latency; NumPy stays the floor.

    Raises
    ------
    RuntimeError
        Explicit comparison is malformed, stale, instrumented or mismatched with
        current runtime/source/configured artifacts. A different CPU uses static order.
    """
    configured = os.environ.get("SC_NEUROCORE_IF_BENCHMARK")
    if not configured:
        return STATIC_ORDER
    try:
        record = json.loads(Path(configured).expanduser().read_text())
        if not isinstance(record, dict):
            raise ValueError("comparison must be an object")
        if (
            record.get("schema_version") != "sc-neurocore.dense-if-comparison.v1"
            or record.get("kernel") != "dense-if-f64-sequential-v1"
        ):
            raise ValueError("comparison schema or kernel is incompatible")
        meta = record["meta"]
        if not isinstance(meta["cpu"], str) or not meta["cpu"]:
            raise ValueError("comparison host CPU must be named")
        cpu = next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        )
        if meta["cpu"] != cpu:
            return STATIC_ORDER
        if (
            meta["python"] != platform.python_version()
            or meta["numpy"] != importlib.metadata.version("numpy")
            or meta["coverage_process_start_requested"] is not False
        ):
            raise ValueError("comparison runtime or instrumentation is incompatible")
        if record["source_sha256"] != source_digests():
            raise ValueError("comparison source binding is stale")
        active = tuple(
            name
            for name in STATIC_ORDER
            if (
                os.environ.get("SC_NEUROCORE_IF_JULIA_ENABLED") == "1"
                if name == "julia"
                else bool(os.environ.get(f"SC_NEUROCORE_IF_{name.upper()}_LIB"))
            )
        )
        if "julia" in active and meta["juliacall"] != importlib.metadata.version("juliacall"):
            raise ValueError("comparison Julia bridge version is incompatible")
        live_artifacts = configured_artifacts(active)
        if any(record["artifact_sha256"].get(key) != sha for key, sha in live_artifacts.items()):
            raise ValueError("comparison native artifact binding is stale")
        return validated_timing_order(record)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError) as error:
        raise RuntimeError(
            "dense IF comparison cannot order this runtime: " + str(error)
        ) from error

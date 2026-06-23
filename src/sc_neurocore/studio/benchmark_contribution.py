# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Community benchmark contribution: schema, local run, databank

"""Run the DCLS kernel benchmark locally and (opt-in) contribute the result.

This is the privacy-controlled contribution path. A user can measure the DCLS
kernel on their own machine and, only if they choose, submit the numbers to a
shared databank so the fastest-backend dispatch can be tuned from real hardware
rather than one reference host.

Privacy is structural, not advisory:

* The submission carries only aggregatable facts — CPU model string, OS family,
  interpreter/toolchain versions, the workload shape and the timing numbers.
* It never carries a hostname, username, IP/MAC address, machine-id or any file
  path; :func:`safe_environment` simply does not collect them and
  :func:`validate_submission` rejects a payload that smuggles a disallowed key.
* Nothing is submitted unless the caller invokes :func:`store_contribution`; the
  contributor handle is optional and free-form.
"""

from __future__ import annotations

import json
import platform
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from sc_neurocore.scpn.dcls_tent_kernel import dcls_max_forward_batch_q88
from sc_neurocore.studio.dcls import probe_backends

SUBMISSION_SCHEMA_VERSION = "scpn.benchmark.submission.v1"
CLIENT_VERSION = "1.0.0"
KERNEL = "dcls_max_forward_batch_q88"

#: The only environment keys a submission may carry. Anything else is rejected.
ALLOWED_ENVIRONMENT_KEYS = frozenset({"cpu", "os", "python", "numpy", "toolchains"})
#: Keys that must never appear anywhere in a submission (machine-identifying).
FORBIDDEN_KEYS = frozenset(
    {"hostname", "host", "node", "user", "username", "ip", "mac", "machine_id", "path"}
)
_HANDLE_RE = re.compile(r"^[\w .\-]{0,40}$")

_DATABANK_FILE = (
    Path(__file__).resolve().parents[3] / "benchmarks" / "databank" / "contributions.jsonl"
)


def safe_environment() -> dict[str, Any]:
    """Collect only the privacy-safe, aggregatable host facts."""

    cpu = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    if cpu == "unknown":
        cpu = platform.processor() or "unknown"
    return {
        "cpu": cpu,  # model string only — never the hostname
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "toolchains": {},  # filled by the live probe; versions only
    }


def _make_workload(
    n_channels: int, n_taps: int, seed: int
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    rng = np.random.default_rng(seed)
    total = n_channels * n_taps
    spikes = (rng.random(total) < 0.5).astype(np.int16)
    weights = rng.integers(-32768, 32768, size=total, dtype=np.int16)
    centres = rng.integers(-256, (n_taps << 8) + 256, size=n_channels, dtype=np.int16)
    sigmas = rng.integers(1, (n_taps << 8) + 256, size=n_channels, dtype=np.int16)
    return spikes, weights, centres, sigmas


def run_local_benchmark(
    n_channels: int = 512, n_taps: int = 32, repeats: int = 12, seed: int = 20260622
) -> dict[str, Any]:
    """Time the DCLS kernel across the in-process-safe backends on this machine.

    Returns a fully-formed submission (schema ``scpn.benchmark.submission.v1``)
    that the caller may inspect and, opt-in, hand to :func:`store_contribution`.
    Julia is skipped in-process (the torch/juliacall segfault) and reported as
    parity-verified offline rather than timed live.
    """

    n_channels = max(16, min(8192, n_channels))
    n_taps = max(4, min(256, n_taps))
    repeats = max(3, min(50, repeats))
    spikes, weights, centres, sigmas = _make_workload(n_channels, n_taps, seed)
    from sc_neurocore.scpn.dcls_tent_kernel import dcls_max_forward_batch

    reference = dcls_max_forward_batch_q88(spikes, weights, centres, sigmas, n_taps)
    ref_out = reference.outputs_q88

    def _median_ms(name: str) -> float:
        dcls_max_forward_batch(spikes, weights, centres, sigmas, n_taps, backend=name)
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            dcls_max_forward_batch(spikes, weights, centres, sigmas, n_taps, backend=name)
            samples.append((time.perf_counter() - start) * 1000.0)
        samples.sort()
        return samples[len(samples) // 2]

    python_ms = _median_ms("python")
    backends: list[dict[str, Any]] = []
    bit_exact_all = True
    for probe in probe_backends():
        name = probe["backend"]
        if not probe["available"] or not probe["live"]:
            continue
        ms = _median_ms(name)
        result = dcls_max_forward_batch(spikes, weights, centres, sigmas, n_taps, backend=name)
        exact = bool(np.array_equal(result.outputs_q88, ref_out))
        bit_exact_all = bit_exact_all and exact
        backends.append(
            {
                "backend": name,
                "median_call_ms": round(ms, 6),
                "channels_per_s": round(n_channels / (ms / 1000.0), 1),
                "speedup_over_python": round(python_ms / ms, 2) if ms > 0 else 0.0,
                "repeats": repeats,
                "bit_exact": exact,
            }
        )
    backends.sort(key=lambda b: b["speedup_over_python"], reverse=True)

    return {
        "schema_version": SUBMISSION_SCHEMA_VERSION,
        "client_version": CLIENT_VERSION,
        "kernel": KERNEL,
        "workload": {
            "n_channels": n_channels,
            "n_taps": n_taps,
            "elements": n_channels * n_taps,
            "spike_density": 0.5,
        },
        "backends": backends,
        "parity": {"reference": "python", "tolerance": 0, "bit_exact_all": bit_exact_all},
        "environment": safe_environment(),
        "hardware_measurement_claimed": False,
        "contributor": {"handle": ""},
    }


def _find_forbidden(node: Any) -> str | None:
    """Walk a payload and return the first machine-identifying key found."""

    if isinstance(node, dict):
        for key, value in node.items():
            if str(key).lower() in FORBIDDEN_KEYS:
                return str(key)
            nested: str | None = _find_forbidden(value)
            if nested is not None:
                return nested
    elif isinstance(node, list):
        for item in node:
            nested = _find_forbidden(item)
            if nested is not None:
                return nested
    return None


def validate_submission(payload: Any) -> list[str]:
    """Return a list of schema/privacy violations; empty means the payload is OK."""

    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["submission must be an object"]
    if payload.get("schema_version") != SUBMISSION_SCHEMA_VERSION:
        errors.append(f"schema_version must be {SUBMISSION_SCHEMA_VERSION!r}")
    if payload.get("kernel") != KERNEL:
        errors.append(f"kernel must be {KERNEL!r}")
    workload = payload.get("workload")
    if not isinstance(workload, dict) or not {"n_channels", "n_taps"} <= set(workload):
        errors.append("workload must declare n_channels and n_taps")
    backends = payload.get("backends")
    if not isinstance(backends, list) or not backends:
        errors.append("backends must be a non-empty list")
    else:
        for entry in backends:
            if (
                not isinstance(entry, dict)
                or "backend" not in entry
                or "median_call_ms" not in entry
            ):
                errors.append("each backend needs a name and median_call_ms")
                break
    environment = payload.get("environment")
    if not isinstance(environment, dict):
        errors.append("environment is required")
    else:
        extra = set(environment) - ALLOWED_ENVIRONMENT_KEYS
        if extra:
            errors.append(f"environment carries disallowed keys: {sorted(extra)}")
    if payload.get("hardware_measurement_claimed") not in (True, False):
        errors.append("hardware_measurement_claimed must be a boolean")
    handle = (
        payload.get("contributor", {}).get("handle", "")
        if isinstance(payload.get("contributor"), dict)
        else None
    )
    if handle is None or not _HANDLE_RE.match(str(handle)):
        errors.append("contributor.handle must be <=40 chars of letters/digits/space/.-_")
    forbidden = _find_forbidden(payload)
    if forbidden is not None:
        errors.append(f"submission must not carry machine-identifying key {forbidden!r}")
    return errors


def store_contribution(payload: dict[str, Any], handle: str = "") -> dict[str, Any]:
    """Validate and append a submission to the local databank (opt-in path).

    Raises ``ValueError`` with the joined violations if the payload fails schema
    or privacy validation, so an invalid or identifying submission never lands.
    """

    payload = dict(payload)
    payload["contributor"] = {"handle": handle.strip()[:40]}
    errors = validate_submission(payload)
    if errors:
        raise ValueError("; ".join(errors))
    _DATABANK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _DATABANK_FILE.open("a", encoding="utf-8") as handle_file:
        handle_file.write(json.dumps(payload, separators=(",", ":")) + "\n")
    return {"stored": True, "schema_version": SUBMISSION_SCHEMA_VERSION}


def load_databank() -> list[dict[str, Any]]:
    """Return every stored contribution (already free of identifying fields)."""

    if not _DATABANK_FILE.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in _DATABANK_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def databank_leaderboard() -> dict[str, Any]:
    """Aggregate the databank into a per-CPU, per-backend speed-up leaderboard."""

    rows = load_databank()
    entries: list[dict[str, Any]] = []
    for row in rows:
        fastest = max(
            (b for b in row.get("backends", [])),
            key=lambda b: b.get("speedup_over_python", 0.0),
            default=None,
        )
        if fastest is None:
            continue
        entries.append(
            {
                "cpu": row.get("environment", {}).get("cpu", "unknown"),
                "handle": row.get("contributor", {}).get("handle", ""),
                "fastest_backend": fastest["backend"],
                "speedup": fastest.get("speedup_over_python", 0.0),
                "workload": row.get("workload", {}),
            }
        )
    entries.sort(key=lambda e: e["speedup"], reverse=True)
    return {"count": len(entries), "entries": entries}

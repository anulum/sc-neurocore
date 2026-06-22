# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio surface for the DCLS-max learnable-delay tent kernel

"""Studio-facing views over the DCLS-max learnable-delay tent kernel.

The catalogue holds single-neuron dynamics; learnable synaptic delays live one
layer down, in the dilated-convolution-with-learnable-spacings (DCLS) tent
kernel. This module surfaces that kernel for the Studio: the learnable tent
weight profile (``centre``/``sigma`` → per-tap triangular gate), a bit-true
forward contraction, and the cross-backend parity that is the kernel's headline
evidence — every acceleration backend agrees with the Python floor bit-for-bit,
because the whole computation is exact integer Q8.8 arithmetic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from sc_neurocore.scpn.dcls_tent_kernel import (
    DEFAULT_FRACTION,
    FASTEST_FIRST_BACKENDS,
    Q88_ONE,
    dcls_max_forward_batch,
    tent_gate_q88,
)


def _julia_unsafe() -> bool:
    """The juliacall bridge segfaults if torch was imported first.

    The Studio loads torch through its neuron models, so a live Julia probe in
    that process would crash (pytorch/pytorch#78829). When torch is present we
    refuse to touch the Julia backend in-process; its bit-exact parity is instead
    covered by the offline ``test_dcls_tent_kernel_parity`` suite.
    """

    return "torch" in sys.modules


def _probe_backend(name: str) -> bool:
    """Probe one backend with a trivial single-tap contraction."""

    dcls_max_forward_batch(
        np.zeros(1, dtype=np.int16),
        np.zeros(1, dtype=np.int16),
        np.zeros(1, dtype=np.int16),
        np.ones(1, dtype=np.int16),
        1,
        backend=name,
    )
    return True


def probe_backends() -> list[dict[str, Any]]:
    """Report each backend's in-process status without ever crashing.

    Each entry is ``{backend, available, live}``: ``live`` is ``False`` for a
    backend we decline to run in this process (Julia under torch), in which case
    ``available`` reflects its declared support rather than a live probe.
    """

    status: list[dict[str, Any]] = []
    for name in FASTEST_FIRST_BACKENDS:
        if name == "python":
            status.append({"backend": name, "available": True, "live": True})
            continue
        if name == "julia" and _julia_unsafe():
            status.append({"backend": name, "available": True, "live": False})
            continue
        try:
            _probe_backend(name)
            status.append({"backend": name, "available": True, "live": True})
        except (ImportError, OSError, RuntimeError, FileNotFoundError):
            status.append({"backend": name, "available": False, "live": True})
    return status

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RTL_FILES = (
    "hdl/sc_dcls_tent_kernel.v",
    "hdl/sc_dcls_axonal_delay.v",
    "hdl/sc_dcls_layer_core.v",
)

#: Hard caps so an interactive request can never ask for an unbounded kernel.
_MAX_TAPS = 256

_BENCH_FILE = _REPO_ROOT / "benchmarks" / "results" / "bench_dcls_tent_kernel.json"


def dcls_benchmark() -> dict[str, Any] | None:
    """Return the recorded multi-backend throughput benchmark, or ``None``.

    These are pre-measured numbers from ``benchmarks/results`` — a CPU-shielded
    run on a known host — not a live timing, so the Studio reports honest,
    reproducible figures with their measurement context rather than noisy
    per-request samples. Backends are ordered fastest-measured-first; Python is
    the 1x reference floor.
    """

    if not _BENCH_FILE.is_file():
        return None
    raw = json.loads(_BENCH_FILE.read_text(encoding="utf-8"))
    raw_backends = raw.get("backends", {})
    rows: list[dict[str, Any]] = []
    for name, entry in raw_backends.items():
        if not entry.get("used"):
            continue
        rows.append(
            {
                "backend": name,
                "median_call_ms": entry.get("median_call_ms"),
                "channels_per_s": entry.get("channels_per_s"),
                "speedup_over_python": entry.get("speedup_over_python", 1.0),
            }
        )
    rows.sort(key=lambda r: r["speedup_over_python"], reverse=True)
    meta = raw.get("meta", {})
    return {
        "date_utc": raw.get("date_utc"),
        "cpu": meta.get("cpu"),
        "workload": raw.get("workload"),
        "isolation_mode": raw.get("benchmark_isolation_mode"),
        "hardware_measurement_claimed": raw.get("hardware_measurement_claimed", False),
        "backends": rows,
    }


def dcls_kernel_info() -> dict[str, Any]:
    """Describe the DCLS-max kernel: provenance, fixed-point contract, evidence."""

    return {
        "name": "DCLS-max learnable-delay tent kernel",
        "provenance": {
            "authors": ["Khalfaoui-Hassani, I.", "Pellegrini, T.", "Masquelier, T."],
            "year": 2023,
            "venue": "ICLR",
            "title": "Dilated convolution with learnable spacings",
            "doi": "10.48550/arXiv.2112.03740",
        },
        "fixed_point": {
            "weight_format": "Q8.8",
            "accumulator_format": "Q16.16",
            "one": Q88_ONE,
            "fraction_bits": DEFAULT_FRACTION,
            "parity": "bit-exact (tolerance 0)",
        },
        "backends": probe_backends(),
        "backend_order": list(FASTEST_FIRST_BACKENDS),
        "rtl_modules": [f for f in _RTL_FILES if (_REPO_ROOT / f).is_file()],
        "synthesis_target": "Xilinx Zynq UltraScale+ ZU3EG",
    }


def dcls_tent_profile(centre_q88: int, sigma_q88: int, n_taps: int) -> dict[str, Any]:
    """Return the per-tap triangular gate profile of a learnable tent kernel.

    Each delay tap ``t`` receives a Q8.8 gate from :func:`tent_gate_q88`; the
    profile is what the learnable ``centre``/``sigma`` shape and what the synapse
    convolves the spike taps against.
    """

    if not 1 <= n_taps <= _MAX_TAPS:
        raise ValueError(f"n_taps must be in 1..{_MAX_TAPS}, got {n_taps}")
    if sigma_q88 <= 0:
        raise ValueError(f"sigma_q88 must be positive, got {sigma_q88}")
    gates = [tent_gate_q88(t, int(centre_q88), int(sigma_q88)) for t in range(n_taps)]
    return {
        "centre_q88": int(centre_q88),
        "sigma_q88": int(sigma_q88),
        "centre": centre_q88 / Q88_ONE,
        "sigma": sigma_q88 / Q88_ONE,
        "n_taps": n_taps,
        "gates_q88": gates,
        "gates": [g / Q88_ONE for g in gates],
    }


def dcls_forward_parity(
    spikes: list[int],
    weights_q88: list[int],
    centre_q88: int,
    sigma_q88: int,
) -> dict[str, Any]:
    """Run the contraction on every available backend and report the parity.

    The Python floor is the reference; each accelerated backend must reproduce
    its Q8.8 output bit-for-bit. The returned ``bit_exact`` flag is the evidence
    that the learnable-delay kernel is hardware-faithful across the whole stack.
    """

    n_taps = len(spikes)
    if n_taps == 0:
        raise ValueError("at least one tap is required")
    if len(weights_q88) != n_taps:
        raise ValueError("spikes and weights must have equal length")
    if sigma_q88 <= 0:
        raise ValueError(f"sigma_q88 must be positive, got {sigma_q88}")

    spike_arr = np.asarray(spikes, dtype=np.int16)
    weight_arr = np.asarray(weights_q88, dtype=np.int16)
    centres = np.asarray([centre_q88], dtype=np.int16)
    sigmas = np.asarray([sigma_q88], dtype=np.int16)

    reference = dcls_max_forward_batch(
        spike_arr, weight_arr, centres, sigmas, n_taps, backend="python"
    )
    ref_out = int(reference.outputs_q88[0])

    per_backend: list[dict[str, Any]] = []
    for probe in probe_backends():
        name = probe["backend"]
        if not probe["available"]:
            per_backend.append({"backend": name, "available": False, "live": True})
            continue
        if not probe["live"]:
            # Declared-supported but not run in-process (Julia under torch);
            # its parity is asserted by the offline parity test suite.
            per_backend.append(
                {"backend": name, "available": True, "live": False, "parity": "offline"}
            )
            continue
        result = dcls_max_forward_batch(
            spike_arr, weight_arr, centres, sigmas, n_taps, backend=name
        )
        out = int(result.outputs_q88[0])
        per_backend.append(
            {
                "backend": name,
                "available": True,
                "live": True,
                "output_q88": out,
                "output": out / Q88_ONE,
                "bit_exact": out == ref_out,
            }
        )

    return {
        "reference_output_q88": ref_out,
        "reference_output": ref_out / Q88_ONE,
        "active_tap_count": int(reference.active_tap_counts[0]),
        "max_gate_q88": int(reference.max_gates_q88[0]),
        "overflow": bool(reference.overflow[0]),
        "backends": per_backend,
        "bit_exact": all(
            b["bit_exact"] for b in per_backend if b.get("live") and b.get("available")
        ),
    }

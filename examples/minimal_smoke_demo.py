# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Minimal install smoke demo

"""Dependency-light smoke demo for ``pip install sc-neurocore[minimal]``."""

from __future__ import annotations

from dataclasses import dataclass
import json
import sys
import time

from sc_neurocore import (
    RNG,
    StochasticLIFNeuron,
    bitstream_to_probability,
    generate_bernoulli_bitstream,
)
from sc_neurocore.hdl import baseline_primitive_text, list_baseline_primitive_rtl

_HEAVY_OPTIONAL_MODULES: tuple[str, ...] = (
    "torch",
    "jax",
    "qiskit",
    "pennylane",
    "lava",
    "gdsfactory",
)


@dataclass(frozen=True)
class MinimalSmokeResult:
    """Structured result for the dependency-light smoke path."""

    bitstream_probability: float
    spike_count: int
    hdl_primitive_count: int
    hdl_sample_module: str
    heavy_modules_loaded: tuple[str, ...]
    elapsed_seconds: float


def _loaded_heavy_modules() -> tuple[str, ...]:
    """Return opt-in dependency roots that were loaded by the current process."""

    return tuple(module for module in _HEAVY_OPTIONAL_MODULES if module in sys.modules)


def run_smoke_demo() -> MinimalSmokeResult:
    """Run the minimal profile smoke path and return measured evidence."""

    started = time.perf_counter()
    bitstream = generate_bernoulli_bitstream(0.62, 256, RNG(seed=20260703))
    neuron = StochasticLIFNeuron(
        v_threshold=1.0,
        tau_mem=10.0,
        noise_std=0.0,
        refractory_period=0,
        seed=7,
    )
    spikes = neuron.process_bitstream(bitstream, input_scale=0.35)
    primitive_names = list_baseline_primitive_rtl()
    sample_module = primitive_names[0]
    primitive_text = baseline_primitive_text(sample_module)
    if f"module {sample_module.removesuffix('.v')}" not in primitive_text:
        raise RuntimeError(f"Unexpected packaged HDL primitive content: {sample_module}")

    return MinimalSmokeResult(
        bitstream_probability=bitstream_to_probability(bitstream),
        spike_count=int(spikes.sum()),
        hdl_primitive_count=len(primitive_names),
        hdl_sample_module=sample_module,
        heavy_modules_loaded=_loaded_heavy_modules(),
        elapsed_seconds=time.perf_counter() - started,
    )


def smoke_payload() -> dict[str, object]:
    """Return the smoke result as a JSON-serializable dictionary."""

    result = run_smoke_demo()
    return {
        "bitstream_probability": result.bitstream_probability,
        "spike_count": result.spike_count,
        "hdl_primitive_count": result.hdl_primitive_count,
        "hdl_sample_module": result.hdl_sample_module,
        "heavy_modules_loaded": list(result.heavy_modules_loaded),
        "elapsed_seconds": result.elapsed_seconds,
    }


def main() -> dict[str, object]:
    """Print and return the minimal smoke result."""

    payload = smoke_payload()
    print(json.dumps(payload, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()

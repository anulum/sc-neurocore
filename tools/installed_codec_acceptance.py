#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Acceptance of installed codec wheels

"""Qualify wheel identity and every installed public codec variant outside the checkout."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter_ns
from typing import TYPE_CHECKING
from zipfile import ZipFile

if TYPE_CHECKING:
    from installed_codec_cases import Case, Output


def _hash(data: bytes) -> str:
    """Return the fingerprint of a transmitted packet, program, or wheel."""
    return hashlib.sha256(data).hexdigest()


def _origin(
    name: str, distribution: importlib.metadata.Distribution, package: ZipFile, checkout: Path
) -> dict[str, str]:
    """Require a real installed file owned by the distribution and identical to its wheel member."""
    module = importlib.import_module(name)
    filename = module.__file__
    if filename is None:
        raise ValueError(f"{name} has no installed file")
    path = Path(filename).resolve()
    root = Path(str(distribution.locate_file(""))).resolve()
    if path.is_relative_to(checkout) or not path.is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError(f"{name} did not load from this installation")
    relative = path.relative_to(root).as_posix()
    owned = {record.as_posix() for record in distribution.files or ()}
    if relative not in owned:
        raise ValueError(f"{relative} is not owned by the installed distribution")
    digest = _hash(path.read_bytes())
    if digest != _hash(package.read(relative)):
        raise ValueError(f"{relative} does not match the supplied wheel")
    return {"member": relative, "sha256": digest}


def _exports(checkout: Path) -> set[str]:
    """Resolve the installed package's advertised entry points and owning modules."""
    from sc_neurocore import spike_codec
    from sc_neurocore.spike_codec import list_codec_descriptors

    names = {"sc_neurocore", spike_codec.__name__}
    listed = set(spike_codec.__all__)
    for descriptor in list_codec_descriptors():
        for entrypoint in (descriptor.encoder_entrypoint, descriptor.decoder_entrypoint):
            path, _, symbol = entrypoint.rpartition(".")
            if path != spike_codec.__name__ or symbol not in listed:
                raise ValueError(f"{entrypoint} is not a public package export")
            target = getattr(importlib.import_module(path), symbol)
            if not callable(target):
                raise ValueError(f"{entrypoint} is not callable")
            names.add(target.__module__)
    if importlib.util.find_spec("sc_neurocore.world_model") is not None:
        raise ValueError(
            "the standard base distribution unexpectedly contains the research package"
        )
    if Path(spike_codec.__file__).resolve().is_relative_to(checkout):
        raise ValueError("codec package came from the checkout")
    return names | {"sc_neurocore.spike_codec._world_model"}


def _installation(
    checkout: Path, base: Path, engine: Path | None, profile: str
) -> dict[str, object]:
    """Bind the imported files, declared version floor and compiled ABI to actual wheel bytes."""
    distribution = importlib.metadata.distribution("sc-neurocore")
    with ZipFile(base) as package:
        modules = {
            name: _origin(name, distribution, package, checkout)
            for name in sorted(_exports(checkout))
        }
    record: dict[str, object] = {
        "base_version": distribution.version,
        "requires_python": distribution.metadata["Requires-Python"],
        "base_wheel": base.name,
        "base_sha256": _hash(base.read_bytes()),
        "base_modules": modules,
        "engine": None,
    }
    if engine is None:
        if profile != "python" or importlib.util.find_spec("sc_neurocore_engine") is not None:
            raise ValueError("the Python-only installation must not contain an engine")
        return record
    if profile == "python":
        raise ValueError("the Python-only profile does not accept an engine wheel")
    native = importlib.metadata.distribution("sc-neurocore-engine")
    with ZipFile(engine) as package:
        facade = _origin("sc_neurocore_engine", native, package, checkout)
        binding = _origin("sc_neurocore_engine.sc_neurocore_engine", native, package, checkout)
    if not binding["member"].endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES)):
        raise ValueError("native codec symbols did not come from a compiled extension")
    implementation = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")
    if implementation.__version__ != native.version:
        raise ValueError("the engine's compiled version differs from its wheel metadata")
    if profile == "native" and (
        distribution.version != native.version
        or distribution.metadata["Requires-Python"] != native.metadata.get("Requires-Python")
    ):
        raise ValueError("the matched base and engine wheels declare incompatible versions")
    record["engine"] = {
        "version": native.version,
        "requires_python": native.metadata.get("Requires-Python"),
        "wheel": engine.name,
        "sha256": _hash(engine.read_bytes()),
        "facade": facade,
        "binding": binding,
    }
    return record


def _match(expected: Output, actual: Output) -> None:
    """Check dtype, shape and exact elements in the complete consumer output."""
    import numpy as np

    for left, right in zip(expected, actual, strict=True):
        if left is None or right is None:
            if left is not right:
                raise ValueError("consumer omitted a decoded field")
        elif left.dtype != right.dtype or not np.array_equal(left, right):
            raise ValueError("consumer output differs across backends")


def _timings(case: Case, repetitions: int) -> dict[str, list[int]]:
    """Measure the two qualified round trips in alternating order after warmup."""
    python: list[int] = []
    rust: list[int] = []
    lanes = (("python", python), ("rust", rust))
    for index in range(repetitions):
        for backend, durations in lanes if index % 2 == 0 else reversed(lanes):
            start = perf_counter_ns()
            case.run(backend, None)
            durations.append(perf_counter_ns() - start)
    return {"python": python, "rust": rust}


def _accept(case: Case, profile: str, repetitions: int) -> dict[str, object]:
    """Exercise auto and explicit backends and cross-read their original packets."""
    from sc_neurocore.spike_codec import probe_codec

    available = probe_codec(case.family, case.variant)
    if not available.python_available or "auto" not in available.backend_choices:
        raise ValueError(f"unusable package: {available.diagnostics}")
    python = case.run("python", None)
    if (python.backend, python.decoder_backend) != ("python", "python"):
        raise ValueError("explicit Python did not select the reference backend")
    automatic = case.run("auto", None)
    if profile == "native" and not available.rust_components_available:
        raise ValueError(f"native wheel lacks the variant: {available.diagnostics}")
    if profile == "python" and (
        automatic.backend != "python" or automatic.decoder_backend != "python"
    ):
        raise ValueError("automatic selection did not use the Python reference")
    if available.rust_components_available:
        rust = case.run("rust", None)
        if (rust.backend, rust.decoder_backend) != ("rust", "rust"):
            raise ValueError("explicit Rust did not select both native endpoints")
        if (automatic.backend, automatic.decoder_backend) != ("rust", "rust"):
            raise ValueError("automatic selection did not use the qualified native backend")
        if rust.packets != python.packets:
            raise ValueError("explicit native packet bytes differ from the Python reference")
        _match(python.output, rust.output)
        _match(python.output, case.run("rust", python.packets).output)
        _match(python.output, case.run("python", rust.packets).output)
    else:
        try:
            case.run("rust", python.packets)
        except ValueError as error:
            if "backend 'rust' needs" not in str(error):
                raise
        else:
            raise ValueError("an unavailable explicit Rust path was accepted")
    _match(python.output, case.run("auto", python.packets).output)
    _match(python.output, case.run("python", automatic.packets).output)
    if automatic.packets != python.packets:
        raise ValueError("automatic packet bytes differ from the installed Python reference")
    return {
        "codec": case.family,
        "variant": case.variant,
        "workload": case.workload,
        "encoder_backend": automatic.backend,
        "decoder_backend": automatic.decoder_backend,
        "native_scope": automatic.native_scope
        or (available.native_scope if available.rust_components_available else "none"),
        "probe_choices": available.backend_choices,
        "packets": len(automatic.packets),
        "packet_bytes": sum(map(len, automatic.packets)),
        "packet_sha256": _hash(
            b"".join(len(packet).to_bytes(8, "big") + packet for packet in automatic.packets)
        ),
        "roundtrip_ns": _timings(case, repetitions) if repetitions else None,
    }


def _benchmark(checkout: Path, processor: str) -> dict[str, object]:
    """Record the measured host, thread limits, and source inputs beside raw times."""
    from numpy import __version__ as numpy_version
    from scipy import __version__ as scipy_version

    files = [
        checkout / name
        for name in (
            "Cargo.lock",
            "Cargo.toml",
            "pyproject.toml",
            "engine/Cargo.toml",
            "engine/src/bindings/isi.rs",
            "engine/src/bindings/delta.rs",
            "engine/src/bindings/predictive_coding.rs",
            "engine/src/address_events.rs",
            "engine/src/raster_stream.rs",
            "engine/src/predictive_coding.rs",
            "bridge/pyproject.toml",
            "bridge/sc_neurocore_engine/__init__.py",
            "tools/installed_codec_acceptance.py",
            "tools/installed_codec_cases.py",
        )
    ]
    for directory, pattern in (
        ("src/sc_neurocore/spike_codec", "*.py"),
        ("engine/src/codecs", "*.rs"),
        ("engine/src/address_events", "*.rs"),
        ("engine/src/raster_stream", "*.rs"),
        ("engine/src/predictive_coding", "*.rs"),
        ("engine/src", "spike_events.rs"),
        ("engine/src", "waveform*.rs"),
        ("engine/src/bindings", "*codec*.rs"),
        ("engine/src/bindings", "spike_events.rs"),
        ("engine/src/bindings", "waveform*.rs"),
    ):
        files.extend((checkout / directory).rglob(pattern))
    return {
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "processor": processor,
        "logical_cpus": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "load_averages": os.getloadavg() if hasattr(os, "getloadavg") else None,
        "threads": {
            key: os.environ.get(key)
            for key in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "RAYON_NUM_THREADS",
            )
        },
        "numpy": numpy_version,
        "scipy": scipy_version,
        "zstandard": _optional_version("zstandard"),
        "pywavelets": _optional_version("PyWavelets"),
        "source_hashes": {
            path.relative_to(checkout).as_posix(): _hash(path.read_bytes())
            for path in sorted(set(files))
        },
    }


def _optional_version(name: str) -> str | None:
    """Report an optional compression dependency when it is installed."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    """Run the installed consumer and write a wheel-bound receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--base-wheel", type=Path, required=True)
    parser.add_argument("--engine-wheel", type=Path)
    parser.add_argument("--profile", choices=("python", "native", "legacy"), required=True)
    parser.add_argument("--recording", type=Path)
    parser.add_argument("--repetitions", type=int, default=0)
    parser.add_argument("--processor", help="measured host CPU model for the benchmark receipt")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    if not sys.flags.isolated or Path.cwd().resolve().is_relative_to(checkout):
        parser.error("run with python -I from outside the checkout")
    if args.repetitions < 0 or (
        args.repetitions and (args.profile != "native" or not args.processor)
    ):
        parser.error("paired timings need a native profile, positive repetitions and a processor")

    # Only the two verification scripts come from the checkout.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from installed_codec_cases import cases

    from sc_neurocore.spike_codec import list_codec_descriptors

    installed = _installation(checkout, args.base_wheel, args.engine_wheel, args.profile)
    plan = cases(args.recording)
    described = {
        (item.name, variant.name) for item in list_codec_descriptors() for variant in item.variants
    }
    if {(case.family, case.variant) for case in plan} != described:
        raise ValueError("the installed acceptance cases no longer cover the catalogue")

    reports = []
    for case in plan:
        try:
            reports.append(_accept(case, args.profile, args.repetitions))
        except (ImportError, OSError, RuntimeError, ValueError, AssertionError) as error:
            raise RuntimeError(f"{case.family}/{case.variant}/{case.workload}: {error}") from error
    report = {
        "schema": 1,
        "profile": args.profile,
        "platform": platform.platform(),
        "python": sys.version,
        "abi": sys.implementation.cache_tag,
        "recording": args.recording.name if args.recording else None,
        "input_sha256": _hash(args.recording.read_bytes()) if args.recording else None,
        "installation": installed,
        "cases": reports,
    }
    if args.repetitions:
        report.update(_benchmark(checkout, args.processor))
        report["repetitions"] = args.repetitions
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"{len(described)} installed variants, {len(reports)} verified workloads ({args.profile})"
    )


if __name__ == "__main__":
    main()

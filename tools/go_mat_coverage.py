# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Complete native Go MAT statement promotion gate

"""Build real native images and qualify all four maintained Go MAT sources.

Run inside an admitted Linux CPU allocation. This tool neither reserves a host
nor changes cooling. Mojo images must already be built from this checkout in
the same CI job; missing images are a refusal. Raw counters remain unfiltered.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import importlib
import inspect
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
from types import FrameType
from typing import NoReturn, Protocol, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.go_coverage_images import running_interpreter_identity
from tools.go_coverage_environment import declare_terminal_environment
from tools.go_coverage_records import snapshot_directories
from tools.go_coverage_profiles import diagnostic_statements
from tools.go_coverage_trace import write_bootstrap
from tools.go_mat_coverage_build import NativeGo, digest, save
from tools.go_coverage_controls import (
    image_refusal_controls,
    native_refusal_controls,
    profile_verdict,
)
from tools.go_coverage_gate_inputs import (
    validate_before_work_sources,
    validate_bootstrap_custody,
    validate_compiler_version,
    validate_final_images,
    validate_image_transfer,
    validate_source_custody,
)


class ReferenceModel(Protocol):
    """Describe the actual Python model surface consumed by the native oracle."""

    v: float
    step: Callable[[float], int]


def source_inputs(repository: Path, go: Path, python_coverage_config: Path) -> dict[str, str]:
    """Freeze potential first-party code inputs without reading installed cache trees."""
    result = {}
    suffixes = {".py", ".go", ".h", ".c", ".s", ".mojo", ".toml", ".mod", ".sum"}
    for directory in (
        repository / "src",
        repository / "tools",
        repository / "tests/native_go_coverage_contracts",
    ):
        for root, dirs, files in os.walk(directory):
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith((".venv", ".pixi"))
                and d not in {"__pycache__", ".git", "node_modules"}
            ]
            for name in files:
                path = Path(root) / name
                if path.suffix in suffixes:
                    result[str(path)] = digest(path)
    for path in (
        repository / "tests/test_mat_family_c_abi.py",
        repository / "tests/native_go_coverage_consumer.py",
        repository / "tests/__init__.py",
        repository / "pyproject.toml",
        repository / "MANIFEST.in",
        repository / "README.md",
        repository / "LICENSE",
        repository / "requirements/build.txt",
        repository / "requirements/ci-dev.txt",
        python_coverage_config,
        python_coverage_config.parent / "source-platform.json",
        go,
        go.parent / "gofmt",
        Path(sys.executable).resolve(strict=True),
    ):
        result[str(path)] = digest(path)
    return result


def python_environment(
    job: Path, repository: Path, phase: str, manifest: str, jobs: int
) -> dict[str, str]:
    """Declare the entire gate-owned consumer mapping, never inheriting vendor variables."""
    if jobs < 1:
        raise ValueError("The declared native runtime concurrency must be positive")
    mapping = {
        **declare_terminal_environment(job),
        "PATH": os.defpath,
        "COVERAGE_RUN": "true",
        "GOMAXPROCS": str(jobs),
        "HOME": os.environ["HOME"],
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LC_CTYPE": "C.UTF-8",
        "TZ": "UTC",
        "TMPDIR": str(job / "tmp"),
        "XDG_CACHE_HOME": str(job / "cache"),
        "GOCOVERDIR": str(job / "automatic-counters"),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "RAYON_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONHASHSEED": "0",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONPATH": str(job / "bootstrap") + os.pathsep + str(repository),
        "NEUROCORE_GO_COVERAGE_JOB": str(job),
        "NEUROCORE_GO_COVERAGE_PHASE": phase,
        "NEUROCORE_GO_COVERAGE_MANIFEST": manifest,
    }
    path = job / "environments" / (phase + ".json")
    if path.exists():
        if json.loads(path.read_text()) != mapping:
            raise RuntimeError("A previously declared consumer environment changed")
    else:
        save(path, mapping)
    return mapping


def prepare_subject(job: Path, repository: Path, built: dict[str, Path]) -> list[dict[str, str]]:
    """Project the unchanged full ABI module onto actual newly declared native images."""
    subject = job / "subject"
    (subject / "tests").mkdir(parents=True)
    for name in ("__init__.py", "test_mat_family_c_abi.py"):
        shutil.copyfile(repository / "tests" / name, subject / "tests" / name)
    images = []
    for model in ("mat", "sc_resetting_mat"):
        for backend in ("go", "mojo"):
            if backend == "go":
                actual = built["instrumented/" + model]
            else:
                original = (
                    repository / "src/sc_neurocore/accel/mojo" / model / ("lib" + model + ".so")
                )
                before = digest(original)
                actual = job / "artifacts/mojo" / model / original.name
                actual.parent.mkdir(parents=True)
                shutil.copyfile(original, actual)
                validate_image_transfer(original, actual, before)
            actual = actual.resolve(strict=True)
            alias = subject / "src/sc_neurocore/accel" / backend / model / actual.name
            alias.parent.mkdir(parents=True)
            alias.symlink_to(actual)
            images.append(
                {"path": str(actual), "sha256": digest(actual), "backend": backend, "model": model}
            )
    save(job / "images.json", images)
    return images


def write_oracle(repository: Path, output: Path) -> None:
    """Run the actual independent Python models over their complete retained workloads."""
    sys.path.insert(0, str(repository / "src"))
    result: dict[str, object] = {}
    for name, class_name, steps, current in (
        ("mat", "MATNeuron", 8192, 0.5),
        ("sc_resetting_mat", "SCResettingMATNeuron", 256, 50.0),
    ):
        module = importlib.import_module("sc_neurocore.neurons.models." + name)
        factory = cast(type[ReferenceModel], getattr(module, class_name))
        expected = repository / "src/sc_neurocore/neurons/models" / (name + ".py")
        if Path(inspect.getfile(factory)).resolve(strict=True) != expected.resolve(strict=True):
            raise RuntimeError("Actual Python oracle class came from another source")
        model = factory()
        advance = model.step
        trace = []
        events = 0
        for _ in range(steps):
            events += advance(current)
            trace.append(float(model.v))
        result[name] = {"steps": steps, "current": current, "voltages": trace, "events": events}
    save(output, result)


def execute(
    repository: Path,
    job: Path,
    go: Path,
    seconds: int,
    jobs: int,
    stop_seconds: float,
    python_coverage_config: Path,
) -> Path:
    """Run the complete native producer and unchanged consumers without host admission effects."""
    if sys.platform != "linux" or sys.flags.optimize or seconds < 1 or jobs < 1:
        raise ValueError("Unoptimized Linux Python and explicit positive budgets are required")
    if job.is_relative_to(repository) or "," in str(job):
        raise ValueError(
            "Use a fresh external output allocation without native input-list separators"
        )
    job.mkdir()
    for name in ("tmp", "cache", "automatic-counters", "environments", "bootstrap"):
        (job / name).mkdir()
    environment = {
        "PATH": str(go.parent) + os.pathsep + os.defpath,
        "HOME": os.environ["HOME"],
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TMPDIR": str(job / "tmp"),
        "GOENV": "off",
        "GOWORK": "off",
        "GOTOOLCHAIN": "local",
        "GOPROXY": "off",
        "GOSUMDB": "off",
        "GOFLAGS": "-mod=readonly",
        "CGO_ENABLED": "1",
        "GOCACHE": str(job / "go-cache"),
        "GOMODCACHE": str(job / "go-mod-cache"),
        "GOMAXPROCS": str(jobs),
    }
    builder = NativeGo(
        job,
        repository,
        go,
        time.monotonic() + seconds,
        jobs,
        environment,
        stop_seconds,
        python_coverage_config,
    )
    save(
        job / "producer-inputs.json",
        {
            "repository": str(repository),
            "go": str(go),
            "deadline_monotonic": builder.deadline,
            "seconds": seconds,
            "started_epoch": time.time(),
            "jobs": jobs,
            "stop_seconds": stop_seconds,
            "environment": environment,
            "resource_admission_claimed": False,
            "python_coverage_config": str(python_coverage_config),
            "python_coverage_sha256": builder.python_coverage_sha256,
        },
    )
    pins = source_inputs(repository, go, python_coverage_config)
    save(job / "source-before.json", pins)
    save(job / "python.json", running_interpreter_identity())
    builder.require("go-version", [str(go), "version"], builder.source)
    observed = (job / "commands/go-version/stdout.log").read_text()
    validate_compiler_version((builder.source / "go.mod").read_text(), observed)
    built = builder.build_images()
    images = prepare_subject(job, repository, built)
    preload = write_bootstrap(job, repository, python_coverage_config)
    save(job / "bootstrap-before.json", {p.name: digest(p) for p in (job / "bootstrap").iterdir()})
    before_env = python_environment(job, repository, "before", "images.json", jobs)
    builder.require(
        "before-work-images",
        builder.python_script(preload, [str(repository), str(job), "images.json"]),
        job,
        environment=before_env,
    )
    before_dirs = snapshot_directories(job, "before")
    before = builder.profiles(before_dirs, "before")
    baseline_verdict = profile_verdict(builder, before, before, "before-work-verdict", False)
    baseline_rows = json.loads(baseline_verdict.read_text())["observed_after"]["sources"]
    validate_before_work_sources(baseline_rows)
    oracle = job / "artifacts/python-oracle.json"
    write_oracle(repository, oracle)
    native = builder.test_family(oracle)
    consumer_env = python_environment(job, repository, "tests", "images.json", jobs)
    consumer = builder.python_script(
        repository / "tests/native_go_coverage_consumer.py",
        [
            "--job",
            str(job),
            "--repository",
            str(repository),
        ],
    )
    builder.require(
        "collect-complete-abi",
        consumer + ["--collect-only"],
        job,
        environment=consumer_env,
        outputs=(job / "collection.json", job / "collection-process.json"),
    )
    builder.require(
        "execute-complete-abi",
        consumer,
        job,
        environment=consumer_env,
        outputs=(job / "execution.json", job / "abi.xml"),
    )
    complete = builder.profiles([*snapshot_directories(job, "tests"), native], "complete")
    verdict = profile_verdict(builder, before, complete, "complete-statement-verdict", True)
    diagnostics = diagnostic_statements(complete)
    save(job / "artifacts/diagnostic-statements.json", diagnostics)
    native_refusal_controls(builder, before, complete, before_dirs[0])
    image_refusal_controls(
        builder,
        images,
        built,
        preload,
        lambda phase, manifest: python_environment(job, repository, phase, manifest, jobs),
    )
    contracts_environment = python_environment(job, repository, "contracts", "images.json", jobs)
    builder.require(
        "dedicated-gate-contracts",
        builder.python_script(
            repository / "tests/native_go_coverage_contracts/run.py",
            [
                "--job",
                str(job),
                "--repository",
                str(repository),
            ],
        ),
        job,
        environment=contracts_environment,
        outputs=(job / "contracts.xml",),
    )
    validate_final_images(images)
    save(job / "bootstrap-after.json", {p.name: digest(p) for p in (job / "bootstrap").iterdir()})
    validate_bootstrap_custody(
        (job / "bootstrap-before.json").read_bytes(), (job / "bootstrap-after.json").read_bytes()
    )
    save(job / "source-after.json", source_inputs(repository, go, python_coverage_config))
    validate_source_custody(pins, json.loads((job / "source-after.json").read_text()))
    save(
        job / "result.json",
        {
            "passed": True,
            "verdict": str(verdict),
            "images": images,
            "original_abi_cases": 630,
            "native_family_cases": 13,
            "diagnostic_statements": diagnostics,
            "source_pins_unchanged": True,
            "resource_admission_claimed": False,
            "branch_or_mcdc_qualification": False,
            "all_function_qualification": False,
        },
    )
    return job / "result.json"


def _cancel(signum: int, frame: FrameType | None) -> NoReturn:
    """Interrupt this CLI so the owned command's finally block reaps its process group."""
    raise InterruptedError("Native gate interrupted by " + signal.Signals(signum).name)


def main() -> int:
    """Require explicit source, output, real compiler and already-admitted work budgets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--go", type=Path, required=True)
    parser.add_argument("--seconds", type=int, required=True)
    parser.add_argument("--jobs", type=int, required=True)
    parser.add_argument("--stop-seconds", type=float, required=True)
    parser.add_argument("--python-coverage-config", type=Path, required=True)
    args = parser.parse_args()
    previous = (
        [(name, signal.getsignal(name)) for name in (signal.SIGALRM, signal.SIGTERM)]
        if sys.platform == "linux" and args.seconds > 0
        else []
    )
    if previous:
        for name, _ in previous:
            signal.signal(name, _cancel)
        signal.setitimer(signal.ITIMER_REAL, args.seconds)

    def restore_signals() -> None:
        """Restore only handlers and the timer installed by this actual invocation."""
        if previous:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for name, handler in previous:
                signal.signal(name, handler)

    try:
        output = execute(
            args.repository.resolve(strict=True),
            args.output.resolve(),
            args.go.resolve(strict=True),
            args.seconds,
            args.jobs,
            args.stop_seconds,
            args.python_coverage_config.resolve(strict=True),
        )
    except (OSError, ValueError, RuntimeError, TimeoutError) as error:
        print("Native Go MAT gate refused: " + repr(error), file=sys.stderr)
        return 1
    finally:
        restore_signals()
    print(str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

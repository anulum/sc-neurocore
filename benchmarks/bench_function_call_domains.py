# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — First-call and cached function-domain analysis timings
"""Measure source-located function analysis in fresh per-model interpreters.

Imports and authored model construction precede timing. The first collector
call includes its numerical sweeps; subsequent calls can reuse accuracy caches.
Each repeat checks the complete serialised contract outside the timed region.
These are loaded-host local measurements of Python metadata analysis, with no
claim about generated hardware speed or an equivalent native implementation.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import statistics
import subprocess
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks._benchmark_context import load_average, measurement_context
from sc_neurocore.compiler import function_call_domains
from sc_neurocore.compiler.function_call_domains import FunctionCallContract
from sc_neurocore.neurons.equation_builder import EquationNeuron

CASES = (
    "paired_exponentials",
    "clipped_hypot",
    "literal_dynamic_powers",
    "exponential_euler_events",
)
DEFAULT_OUTPUT = ROOT / "benchmarks/results/bench_function_call_domains.json"
SOURCE_PATHS = tuple(
    sorted(
        {
            "benchmarks/bench_function_call_domains.py",
            "benchmarks/_benchmark_context.py",
            *(
                p.relative_to(ROOT).as_posix()
                for p in (ROOT / "src/sc_neurocore/compiler").glob("*.py")
            ),
            *(
                p.relative_to(ROOT).as_posix()
                for p in (ROOT / "src/sc_neurocore/neurons").glob("*.py")
            ),
        }
    )
)


def _source_hashes() -> dict[str, str]:
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCE_PATHS}


def _model(name: str) -> EquationNeuron:
    return {
        "paired_exponentials": EquationNeuron({"v": "exp(I)+exp(I/4)"}, method="map"),
        "clipped_hypot": EquationNeuron({"v": "hypot(clip(I,-1,1),clip(v,-1,1))"}, method="map"),
        "literal_dynamic_powers": EquationNeuron(
            {"v": "I**1.5+clip(I,.25,1)**clip(w,-1,1)"},
            state={"w": 0.5},
            method="map",
        ),
        "exponential_euler_events": EquationNeuron(
            {"v": "-v+exp(I)/4"},
            state={"v": 0.0},
            method="exp_euler",
            dt=0.125,
            threshold="exp(v)>2",
            reset={"v": "tanh(I)"},
            spike_value="sigmoid(v)",
        ),
    }[name]


def _serialise(contracts: tuple[FunctionCallContract, ...]) -> str:
    if not contracts or any(call.refusal or call.accuracy is None for call in contracts):
        raise ValueError("benchmark model has no complete call-domain evidence")
    return json.dumps([call.to_public_dict(4) for call in contracts], sort_keys=True)


def measure_case(name: str, calls: int, repeats: int) -> dict[str, object]:
    """Measure one model's first collector call and cached repetitions.

    Parameters
    ----------
    name : str
        One of the authored model names in ``CASES``.
    calls : int
        Collector invocations per cached repeat.
    repeats : int
        Number of cached repeats.

    Returns
    -------
    dict
        First-call latency, cached samples and the complete measured contract.

    Raises
    ------
    ValueError
        Counts are non-positive, contracts are refused or results change.
    KeyError
        The model name is unknown.
    """
    if min(calls, repeats) <= 0:
        raise ValueError("calls and repeats must be positive")
    model = _model(name)
    started = perf_counter_ns()
    contracts = function_call_domains.function_call_contracts(model, 8, 4)
    first_ns = perf_counter_ns() - started
    expected = _serialise(contracts)
    samples: list[float] = []
    for _ in range(repeats):
        started = perf_counter_ns()
        for _ in range(calls):
            contracts = function_call_domains.function_call_contracts(model, 8, 4)
        samples.append((perf_counter_ns() - started) / calls)
        if _serialise(contracts) != expected:
            raise ValueError("call-domain contracts changed between repetitions")
    return {
        "first_call_ns": first_ns,
        "cached_ns_per_call": samples,
        "cached_median_ns_per_call": statistics.median(samples),
        "contract_sha256": hashlib.sha256(expected.encode()).hexdigest(),
        "contracts": json.loads(expected),
        "all_repeats_equal": True,
    }


def build_payload(calls: int, repeats: int) -> dict[str, object]:
    """Run each model in its own interpreter and bind timings to source bytes.

    Parameters
    ----------
    calls : int
        Collector calls per cached repeat in each child process.
    repeats : int
        Cached repeats per model.

    Returns
    -------
    dict
        Source-bound receipt with hardware context and distinct cache scopes.

    Raises
    ------
    ValueError
        Counts are non-positive or measured sources change during capture.
    RuntimeError
        The imported collector does not come from this checkout.
    subprocess.CalledProcessError
        A child collector or Git provenance command fails.
    subprocess.TimeoutExpired
        A child model exceeds its ten-minute limit.
    """
    if min(calls, repeats) <= 0:
        raise ValueError("calls and repeats must be positive")
    if (
        Path(function_call_domains.__file__).resolve()
        != (ROOT / "src/sc_neurocore/compiler/function_call_domains.py").resolve()
    ):
        raise RuntimeError("collector import does not belong to this checkout")
    before = _source_hashes()
    load_before = load_average()
    cases: dict[str, object] = {}
    for name in CASES:
        process = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--measure-case",
                name,
                "--calls",
                str(calls),
                "--repeats",
                str(repeats),
            ],
            check=True,
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=600,
        )
        cases[name] = json.loads(process.stdout)
    if _source_hashes() != before:
        raise ValueError("measured sources changed during capture")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "schema_version": "sc-neurocore.function-call-domain-benchmark.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_class": "local_regression",
        "production_speed_claimed": False,
        "language": "Python",
        "configuration": {"calls": calls, "repeats": repeats, "data_width": 8, "fraction": 4},
        "protocol": {
            "first_call": "First collector invocation in a fresh interpreter for each model.",
            "cached": "Later calls in that same process; numerical accuracy caches may be reused.",
            "timed_region": "function_call_contracts only; imports, model creation and JSON excluded.",
            "comparison": "Repeated results checked for exact serialised equality outside timing.",
            "limits": "Shared host; no isolated throughput, hardware speed or native counterpart claim.",
        },
        "measurement_context": measurement_context(load_before),
        "reference_dependency_versions": {
            name: version(name) for name in ("numpy", "scipy", "sympy")
        },
        "source_head_at_capture": head,
        "source_worktree_dirty": bool(dirty),
        "source_sha256": before,
        "source_unchanged_during_capture": True,
        "cases": cases,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Write a full receipt, or measure one case for the parent process.

    Parameters
    ----------
    argv : sequence of str, optional
        CLI arguments; process arguments when omitted.

    Returns
    -------
    int
        Zero after successful measurement and output.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--measure-case", choices=CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.measure_case is not None:
        print(json.dumps(measure_case(args.measure_case, args.calls, args.repeats)))
    else:
        payload = build_payload(args.calls, args.repeats)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

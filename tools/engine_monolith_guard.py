# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Engine crate-root monolith no-growth guard

"""Freeze the engine PyO3 crate root against monolith growth.

The engine crate root ``engine/src/lib.rs`` accreted a new
``py_<neuron>_simulate`` wrapper for every fidelity unit, turning it into the
largest file in the repository while the Python model surfaces were being
decomposed (S-MONO-01…). This guard closes that loop: it holds a **DOWN-only**
ratchet on the crate root's line count and ``#[pyfunction]`` count. A wrapper
appended to ``engine/src/lib.rs`` pushes a count over its frozen ceiling and
fails the check, forcing the binding into ``engine/src/bindings/<neuron>.rs``
(the McCulloch-Pitts pattern) with its own ``register`` function instead.

Usage::

    python tools/engine_monolith_guard.py            # print current vs ceiling
    python tools/engine_monolith_guard.py --check     # CI gate (non-zero on growth)
    python tools/engine_monolith_guard.py --update     # ratchet ceilings down

``--update`` only ever tightens a ceiling; it refuses to raise one, so the
ratchet cannot be loosened to bless new growth.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Iterable

try:  # pragma: no cover - exercised by the Python-version matrix in CI
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

DEFAULT_CEILING = Path("tools/engine_monolith_ceiling.toml")
SCHEMA_VERSION = 1

#: Matches ``#[pyfunction]`` and the argument form ``#[pyfunction(...)]``.
PYFUNCTION_PATTERN = re.compile(r"#\[pyfunction\b")

#: Ceiling key -> measured-metric key.
METRICS: dict[str, str] = {
    "max_lines": "lines",
    "max_pyfunctions": "pyfunctions",
}

_CEILING_HEADER = """\
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Engine crate-root monolith ceiling (DOWN-only ratchet)

# DOWN-only ratchet on the engine PyO3 crate root. Every ceiling here may only
# stay the same or decrease; `tools/engine_monolith_guard.py --update` refuses
# to raise a ceiling. New PyO3 bindings must be added under
# engine/src/bindings/<neuron>.rs with a register() function
# (see engine/src/bindings/mcculloch_pitts.rs), never appended to
# engine/src/lib.rs. See docs/internal/TODO.md — S-MONO-RUST.
"""


class CeilingRaiseError(RuntimeError):
    """Raised when an update would loosen (raise) a frozen ceiling."""


def measure_target(repo: Path, rel_path: str) -> dict[str, int]:
    """Return the line and ``#[pyfunction]`` counts for one tracked file."""

    text = (repo / rel_path).read_text(encoding="utf-8")
    return {
        "lines": len(text.splitlines()),
        "pyfunctions": len(PYFUNCTION_PATTERN.findall(text)),
    }


def load_ceiling(path: Path) -> dict[str, Any]:
    """Load the ceiling document, validating its schema version."""

    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    version = raw.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported ceiling schema_version {version!r}; expected {SCHEMA_VERSION}"
        )
    targets = raw.get("targets")
    if not isinstance(targets, dict) or not targets:
        raise ValueError("ceiling document has no [targets]")
    return raw


def evaluate(repo: Path, ceiling: dict[str, Any]) -> dict[str, Any]:
    """Compare the working tree against the ceiling.

    Returns a report with ``passed``, per-target ``measurements`` and a list of
    ``violations`` (each naming the file, metric, ceiling, actual and delta).
    """

    targets: dict[str, dict[str, int]] = ceiling["targets"]
    measurements: dict[str, dict[str, int]] = {}
    violations: list[dict[str, Any]] = []
    for rel_path in sorted(targets):
        actual = measure_target(repo, rel_path)
        measurements[rel_path] = actual
        for ceiling_key, metric in METRICS.items():
            cap = int(targets[rel_path][ceiling_key])
            got = actual[metric]
            if got > cap:
                violations.append(
                    {
                        "path": rel_path,
                        "metric": metric,
                        "ceiling": cap,
                        "actual": got,
                        "delta": got - cap,
                    }
                )
    return {"passed": not violations, "measurements": measurements, "violations": violations}


def format_report(report: dict[str, Any]) -> str:
    """Render a human-readable one-line-per-target summary."""

    lines = []
    for rel_path, actual in sorted(report["measurements"].items()):
        lines.append(f"{rel_path}: {actual['lines']} lines, {actual['pyfunctions']} #[pyfunction]")
    return "\n".join(lines)


def format_violations(report: dict[str, Any]) -> str:
    """Render an actionable failure message for a failed evaluation."""

    lines = ["engine crate root grew beyond its frozen ceiling:"]
    for violation in report["violations"]:
        lines.append(
            f"  {violation['path']} {violation['metric']}: "
            f"{violation['actual']} > {violation['ceiling']} (+{violation['delta']})"
        )
    lines.append(
        "The engine crate root is frozen DOWN-only (S-MONO-RUST). Add new PyO3 "
        "bindings under engine/src/bindings/<neuron>.rs with a register() "
        "function (see engine/src/bindings/mcculloch_pitts.rs), not in "
        "engine/src/lib.rs. After moving code out, tighten the ceiling with "
        "`python tools/engine_monolith_guard.py --update`."
    )
    return "\n".join(lines)


def tightened_ceiling(repo: Path, ceiling: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``ceiling`` ratcheted down to current measurements.

    Refuses to raise: if the working tree exceeds any ceiling, this raises
    :class:`CeilingRaiseError` so the ratchet cannot be loosened to bless
    growth. Ceilings already at or below the current counts are left untouched.
    """

    targets: dict[str, dict[str, int]] = ceiling["targets"]
    updated_targets: dict[str, dict[str, int]] = {}
    for rel_path in sorted(targets):
        actual = measure_target(repo, rel_path)
        new_target: dict[str, int] = {}
        for ceiling_key, metric in METRICS.items():
            cap = int(targets[rel_path][ceiling_key])
            got = actual[metric]
            if got > cap:
                raise CeilingRaiseError(
                    f"refusing to raise ceiling for {rel_path} {metric}: "
                    f"{got} > {cap}; move bindings out of the crate root first"
                )
            new_target[ceiling_key] = got
        updated_targets[rel_path] = new_target
    return {"schema_version": SCHEMA_VERSION, "targets": updated_targets}


def render_ceiling_toml(ceiling: dict[str, Any]) -> str:
    """Render a ceiling document back to its canonical TOML form."""

    targets: dict[str, dict[str, int]] = ceiling["targets"]
    parts = [_CEILING_HEADER, f"schema_version = {int(ceiling['schema_version'])}\n"]
    for rel_path in sorted(targets):
        target = targets[rel_path]
        parts.append(f'[targets."{rel_path}"]')
        parts.append(f"max_lines = {int(target['max_lines'])}")
        parts.append(f"max_pyfunctions = {int(target['max_pyfunctions'])}\n")
    return "\n".join(parts)


def main(argv: Iterable[str] | None = None) -> int:
    """Entry point: print, check, or ratchet the crate-root ceiling."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--ceiling", type=Path, default=DEFAULT_CEILING)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="fail (non-zero) on growth")
    mode.add_argument("--update", action="store_true", help="ratchet ceilings down to current")
    args = parser.parse_args(list(argv) if argv is not None else None)

    repo = args.repo.resolve()
    ceiling_path = args.ceiling if args.ceiling.is_absolute() else repo / args.ceiling
    ceiling = load_ceiling(ceiling_path)

    if args.update:
        try:
            updated = tightened_ceiling(repo, ceiling)
        except CeilingRaiseError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        ceiling_path.write_text(render_ceiling_toml(updated), encoding="utf-8")
        print(f"Ratcheted {ceiling_path}")
        print(format_report(evaluate(repo, updated)))
        return 0

    report = evaluate(repo, ceiling)
    if args.check:
        if not report["passed"]:
            print(format_violations(report), file=sys.stderr)
            return 1
        print(format_report(report))
        return 0

    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Schema-DSL coverage and co-simulation enrolment report

"""Report schema-DSL coverage for the WC-A5 co-simulation campaign.

The report is derived from source files and schema files in the current
checkout. It does not import neuron modules, so optional backend dependencies
and module side effects cannot affect the output.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


SCHEMA_VERSION = "sc-neurocore.schema-gap-report.v1"
MODEL_DIR = Path("src/sc_neurocore/neurons/models")
SCHEMA_DIR = Path("src/sc_neurocore/neurons/model_schemas")
# Canonical alias table (schema stem ↔ module); keep import local so this tool
# remains usable without a full package install when run as a script.
try:
    from sc_neurocore.neurons.schema_module_aliases import (  # type: ignore[import-not-found]
        SCHEMA_SOURCE_ALIASES as _CANONICAL_SCHEMA_SOURCE_ALIASES,
    )

    SCHEMA_SOURCE_ALIASES: dict[str, str] = dict(_CANONICAL_SCHEMA_SOURCE_ALIASES)
except ImportError:  # pragma: no cover - script path without PYTHONPATH=src
    SCHEMA_SOURCE_ALIASES = {
        "expif": "exp_if",
        "resonate_and_fire": "resonate_fire",
    }

Classification = Literal[
    "schema_present",
    "event_discrete",
    "exact_flow_candidate",
    "euler_candidate",
    "rk4_required",
    "exp_euler_candidate",
    "map_or_discrete",
    "multi_compartment",
    "neural_mass_euler",
    "neural_mass_rk4",
    "stochastic_or_rate",
    "package_alias",
    "source_review_required",
]

PRIORITY_ORDER: dict[str, int] = {
    "P0-schema-present": 0,
    "P1-euler-schema-candidate": 1,
    "P1-exact-flow-schema-candidate": 2,
    "P2-exp-euler-schema-candidate": 3,
    "P3-rk4-or-higher-order-blocked": 4,
    "P4-statistical-or-discrete-validation": 5,
    "P5-out-of-auto-cosim": 6,
    "P6-source-review-required": 7,
}


@dataclass(frozen=True)
class ModelGapRecord:
    """One source-model row in the schema coverage report."""

    model: str
    source_path: str
    schema_present: bool
    schema_name: str | None
    classification: Classification
    priority: str
    evidence: list[str]


def collect_model_names(repo: Path) -> list[str]:
    """Return neuron model module stems, excluding package infrastructure."""
    model_root = repo / MODEL_DIR
    return sorted(path.stem for path in model_root.glob("*.py") if path.name != "__init__.py")


def collect_schema_names(repo: Path) -> list[str]:
    """Return unique schema stems from TOML and JSON schema files."""
    schema_root = repo / SCHEMA_DIR
    toml_names = {path.stem for path in schema_root.glob("*.toml")}
    json_names = {path.stem for path in schema_root.glob("*.json")}
    return sorted(toml_names | json_names)


def build_report(repo: Path) -> dict[str, Any]:
    """Build a JSON-serialisable schema-gap report for ``repo``."""
    repo = repo.resolve()
    schema_names = set(collect_schema_names(repo))
    records = [
        classify_model(repo, model_name, schema_name=schema_for_source(model_name, schema_names))
        for model_name in collect_model_names(repo)
    ]
    missing = [record for record in records if not record.schema_present]
    covered_schema_names = {record.schema_name for record in records if record.schema_name}
    schema_only_models = sorted(schema_names - covered_schema_names)
    class_counts = Counter(record.classification for record in records)
    priority_counts = Counter(record.priority for record in missing)

    return {
        "schema_version": SCHEMA_VERSION,
        "repo": str(repo),
        "counts": {
            "model_modules": len(records),
            "schema_models": len(schema_names),
            "net_missing_schema_models": len(records) - len(schema_names),
            "source_modules_without_schema": len(missing),
            "schema_only_models": len(schema_only_models),
            "classification": dict(sorted(class_counts.items())),
            "missing_priority": dict(sorted(priority_counts.items())),
        },
        "schema_models": sorted(schema_names),
        "schema_only_models": schema_only_models,
        "records": [asdict(record) for record in records],
        "ranked_enrolment": [
            asdict(record)
            for record in sorted(
                missing,
                key=lambda item: (PRIORITY_ORDER[item.priority], item.classification, item.model),
            )
        ],
    }


def classify_model(repo: Path, model: str, *, schema_name: str | None) -> ModelGapRecord:
    """Classify one model source using static source evidence."""
    source_path = MODEL_DIR / f"{model}.py"
    text = (repo / source_path).read_text(encoding="utf-8")
    lower = text.lower()
    schema_present = schema_name is not None

    if schema_present:
        classification: Classification = "schema_present"
        evidence = [f"schema file exists for {schema_name}"]
        priority = "P0-schema-present"
    else:
        classification, evidence = classify_source(model, lower)
        priority = priority_for(classification)

    return ModelGapRecord(
        model=model,
        source_path=source_path.as_posix(),
        schema_present=schema_present,
        schema_name=schema_name,
        classification=classification,
        priority=priority,
        evidence=evidence,
    )


def classify_source(model: str, lower_source: str) -> tuple[Classification, list[str]]:
    """Classify a missing-schema source module from text evidence."""
    if (
        "re-exports" in lower_source
        or "from sc_neurocore." in lower_source
        and "def step" not in lower_source
    ):
        return "package_alias", ["module re-exports another implementation"]
    if _has_any(
        model, lower_source, ("poisson", "stochastic", "random", "rng", "gaussian white noise")
    ):
        return "stochastic_or_rate", ["source contains stochastic/rate-process evidence"]
    if _has_any(
        model,
        lower_source,
        ("compartment", "dendrit", "soma", "purkinje", "pinsky", "hay_l5", "multicompartment"),
    ):
        return "multi_compartment", ["source contains compartmental morphology evidence"]
    if _has_any(
        model,
        lower_source,
        (
            "map",
            "rank-order",
            "integer arithmetic",
            "fixed-point",
            "sigma-delta",
            "buffer",
            "attention",
        ),
    ):
        return "event_discrete", ["source uses discrete/event-domain update evidence"]
    if _has_any(
        model,
        lower_source,
        ("rk4", "runge-kutta", "runge kutta", "k1_", "k2_", "k3_", "k4_", "k1 =", "k2 ="),
    ):
        if _has_any(model, lower_source, ("neural mass", "wilson-cowan", "population rate")):
            return "neural_mass_rk4", ["source uses RK4 neural-mass update evidence"]
        return "rk4_required", ["source uses RK4 / Runge-Kutta evidence"]
    if _has_any(
        model,
        lower_source,
        (
            "exact-flow",
            "exact flow",
            "exact_linear_flow",
            "closed-form",
            "math.exp(-self.dt",
            "np.exp(-self.dt",
        ),
    ):
        return "exact_flow_candidate", ["source contains exact-flow decay evidence"]
    if _has_any(model, lower_source, ("exponential euler", "rush", "exprel", "alpha =")):
        return "exp_euler_candidate", ["source contains exponential-update evidence"]
    if _has_any(model, lower_source, ("neural mass", "eeg", "population", "dy0", "dy1")):
        return "neural_mass_euler", ["source contains neural-mass Euler evidence"]
    if _has_any(
        model, lower_source, ("ode", "dca", "dv", "derivative", "+= ", "* self.dt", "*self.dt")
    ):
        return "euler_candidate", ["source contains explicit-Euler-style update evidence"]
    return "source_review_required", ["no deterministic integrator evidence matched static rules"]


def priority_for(classification: Classification) -> str:
    """Return the WC-A5 enrolment priority bucket for a classification."""
    if classification == "schema_present":
        return "P0-schema-present"
    if classification == "euler_candidate" or classification == "neural_mass_euler":
        return "P1-euler-schema-candidate"
    if classification == "exact_flow_candidate":
        return "P1-exact-flow-schema-candidate"
    if classification == "exp_euler_candidate":
        return "P2-exp-euler-schema-candidate"
    if classification == "rk4_required" or classification == "neural_mass_rk4":
        return "P3-rk4-or-higher-order-blocked"
    if classification in {"event_discrete", "map_or_discrete", "stochastic_or_rate"}:
        return "P4-statistical-or-discrete-validation"
    if classification in {"multi_compartment", "package_alias"}:
        return "P5-out-of-auto-cosim"
    return "P6-source-review-required"


def schema_for_source(model: str, schema_names: set[str]) -> str | None:
    """Return the schema name covering a source model, including known aliases."""
    if model in schema_names:
        return model
    alias = SCHEMA_SOURCE_ALIASES.get(model)
    if alias in schema_names:
        return alias
    return None


def render_markdown(report: dict[str, Any]) -> str:
    """Render a Markdown report from :func:`build_report` output."""
    counts = report["counts"]
    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "<!-- Commercial license available -->",
        "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
        "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
        "<!-- ORCID: 0009-0009-3560-0851 -->",
        "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
        "<!-- SC-NeuroCore — Schema-DSL gap report -->",
        "",
        "# Schema-DSL Gap Report",
        "",
        "This report is generated from the current checkout by",
        "`tools/schema_gap_report.py`. It is a planning aid for WC-A5",
        "Python↔Verilog co-simulation enrolment; it does not certify scientific",
        "validity for any individual model.",
        "",
        "## Counts",
        "",
        f"- Model source modules: **{counts['model_modules']}**",
        f"- Unique schema-DSL models: **{counts['schema_models']}**",
        f"- Net missing schema-DSL models: **{counts['net_missing_schema_models']}**",
        f"- Source modules without a same-name or alias schema: **{counts['source_modules_without_schema']}**",
        f"- Schema-only model names: **{counts['schema_only_models']}**",
        "",
        "## Missing-Schema Priority Buckets",
        "",
        "| Priority | Count | Meaning |",
        "|---|---:|---|",
    ]
    meanings = {
        "P1-euler-schema-candidate": "Direct schema candidate; explicit Euler-style update evidence.",
        "P1-exact-flow-schema-candidate": "Direct schema candidate if the exact flow is preserved or declared.",
        "P2-exp-euler-schema-candidate": "Schema candidate once exponential-Euler semantics are explicit.",
        "P3-rk4-or-higher-order-blocked": "Blocked from parity until higher-order emitter path is ready.",
        "P4-statistical-or-discrete-validation": "Needs statistical/discrete validation, not spike-count parity.",
        "P5-out-of-auto-cosim": "Out of automatic single-schema co-sim path.",
        "P6-source-review-required": "Needs manual source review before enrolment.",
    }
    for priority, count in sorted(
        counts["missing_priority"].items(), key=lambda item: PRIORITY_ORDER[item[0]]
    ):
        lines.append(f"| `{priority}` | {count} | {meanings.get(priority, '')} |")

    lines.extend(
        [
            "",
            "## Ranked Missing Models",
            "",
            "| Priority | Model | Classification | Evidence |",
            "|---|---|---|---|",
        ]
    )
    for record in report["ranked_enrolment"]:
        evidence = "; ".join(record["evidence"])
        lines.append(
            f"| `{record['priority']}` | `{record['model']}` | "
            f"`{record['classification']}` | {evidence} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Parse CLI arguments and write the requested report format."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Repository root")
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    parser.add_argument("--output", type=Path, help="Write report to this path")
    args = parser.parse_args()

    report = build_report(args.repo)
    payload = (
        json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.format == "json"
        else render_markdown(report)
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="" if payload.endswith("\n") else "\n")


def _has_any(model: str, source: str, needles: tuple[str, ...]) -> bool:
    """Return whether any evidence token appears in the model name or source."""
    return any(needle in model or needle in source for needle in needles)


if __name__ == "__main__":
    main()

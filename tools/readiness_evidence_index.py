#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Readiness evidence index (dual-axis facet wiring)

"""Index committed co-simulation evidence and wire dual-axis descriptor facets.

WC-A5 enrols schema models into ``tests/test_cosimulation.py`` with iverilog
and spike-parity gates, but the dual-axis readiness ladder
(:mod:`sc_neurocore.neurons.descriptor_tiers`) only climbs when
:class:`~sc_neurocore.neurons.model_descriptor.Validation` and
:class:`~sc_neurocore.neurons.model_descriptor.Silicon` facets are recorded on
each descriptor. This tool:

1. Inventories a curated, non-overlapping shortlist of **already enrolled**
   models (not new schema enrolments — those stay the WC-A5 agent lane).
2. Reports live science/silicon tiers vs expected evidence.
3. Optionally applies honest facet patches to on-disk descriptors.

Validation is **schema-DSL path** evidence (``UniversalNeuron`` + RTL), not a
claim that every hand ``models/*.py`` class is bit-identical. Operating points
state that boundary explicitly (audit finding P3-02).

Usage::

    PYTHONPATH=src:. python tools/readiness_evidence_index.py --report
    PYTHONPATH=src:. python tools/readiness_evidence_index.py --check
    PYTHONPATH=src:. python tools/readiness_evidence_index.py --apply
    PYTHONPATH=src:. python tools/readiness_evidence_index.py --json docs/internal/readiness_evidence_index.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import tomli_w

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from sc_neurocore.neurons.descriptor_tiers import (  # noqa: E402
    science_tier,
    silicon_tier,
)
from sc_neurocore.neurons.model_descriptor import (  # noqa: E402
    descriptor_completeness_tier,
)
from sc_neurocore.neurons.model_catalogue import (  # noqa: E402
    DESCRIPTOR_DIR,
    descriptor_path,
    load_descriptor,
    load_descriptor_payload,
)

SCHEMA_VERSION = "sc-neurocore.readiness-evidence-index.v1"

# SPDX header shared with the model descriptor corpus generator.
_DESCRIPTOR_HEADER = (
    "# SPDX-License-Identifier: AGPL-3.0-or-later\n"
    "# Commercial license available\n"
    "# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.\n"
    "# © Code 2020–2026 Miroslav Šotek. All rights reserved.\n"
    "# ORCID: 0009-0009-3560-0851\n"
    "# Contact: www.anulum.li | protoscience@anulum.li\n"
    "# SC-NeuroCore — Source/config provenance header\n\n"
)

EvidenceLevel = Literal["h0_compile", "h1_cosim"]


@dataclass(frozen=True, slots=True)
class EnrolledEvidence:
    """One already-enrolled schema model with committed test evidence."""

    schema_name: str
    class_name: str
    level: EvidenceLevel
    evidence: str
    operating_point: str
    tolerance: str
    # Excluded from apply when True (active peer lane / WIP).
    skip_apply: bool = False
    skip_reason: str = ""


# Curated from tests/test_cosimulation.py enrolment (WC-A5). Do not add models
# that still lack a committed cosim/compile gate. Wang-Buzsaki is deliberately
# skip_apply while the peer agent rewrites its integrator (Gauss-Seidel).
ENROLLED: tuple[EnrolledEvidence, ...] = (
    EnrolledEvidence(
        schema_name="cazelles_map",
        class_name="CazellesMapNeuron",
        level="h1_cosim",
        evidence="tests/test_cosim_cazelles_map.py::test_q1616_short_window_trajectory",
        operating_point=("schema-DSL Cazelles map at I=0.5,1.0,2.0 over 30 iterations"),
        tolerance=(
            "hand/TOML/JSON exact; Q16.16 event-exact with state error below "
            "0.0004; I=0.05 excluded"
        ),
    ),
    EnrolledEvidence(
        schema_name="chialvo_map",
        class_name="ChialvoMapNeuron",
        level="h1_cosim",
        evidence=(
            "tests/test_cosim_chialvo_map.py::test_q1616_event_class_and_stable_trajectory_envelope"
        ),
        operating_point=("Chialvo map at I=-0.05,0,0.01,0.1,1.0 over 100 iterations"),
        tolerance=(
            "hand/TOML/JSON exact; Q16.16 event counts exact; stable-point x/y errors "
            "below 0.055/0.093"
        ),
    ),
    EnrolledEvidence(
        schema_name="courage_nekorkin_map",
        class_name="CourageNekorkinMapNeuron",
        level="h1_cosim",
        evidence=("tests/test_cosim_courage_nekorkin_map.py::test_q3232_short_window_trajectory"),
        operating_point=("schema-DSL Courbage-Nekorkin map at I=-0.3,0,0.3 over 30 iterations"),
        tolerance=(
            "hand/TOML/JSON exact; Q32.32 event-exact with state error below "
            "0.00003; Q16.16 autonomous 30-iteration boundary excluded"
        ),
    ),
    EnrolledEvidence(
        schema_name="ermentrout_kopell_map_neuron",
        class_name="ErmentroutKopellMapNeuron",
        level="h1_cosim",
        evidence=(
            "tests/test_cosim_ermentrout_kopell_map_neuron.py::"
            "test_q1616_class_correct_spike_count_and_circular_phase_bound"
        ),
        operating_point=("maintained theta-Euler recurrence over 2000 steps at I=-0.5,0.5,1.0"),
        tolerance=(
            "hand/TOML/JSON exact; Q16.16 spike counts 0/45/64 exact with maximum "
            "circular phase error below 0.081/0.089/0.025 rad"
        ),
    ),
    EnrolledEvidence(
        schema_name="lapicque",
        class_name="LapicqueNeuron",
        level="h1_cosim",
        evidence="tests/test_cosimulation.py::_COSIM_MODELS cosim suite (lapicque)",
        operating_point=(
            "schema-DSL lapicque (sibling schema 'lif' shares the same class "
            "descriptor until a dedicated StochasticLIF descriptor exists)"
        ),
        tolerance="suite parity gate",
    ),
    EnrolledEvidence(
        schema_name="perfect_integrator",
        class_name="PerfectIntegratorNeuron",
        level="h1_cosim",
        evidence="tests/test_cosimulation.py perfect_integrator enrolment",
        operating_point="schema-DSL perfect_integrator; hand/schema/RTL path",
        tolerance="exact or suite-defined band",
    ),
    EnrolledEvidence(
        schema_name="quadratic_if",
        class_name="QuadraticIFNeuron",
        level="h1_cosim",
        evidence="tests/test_cosimulation.py::_RK4_EXACT_MODELS + _COSIM_MODELS",
        operating_point="schema-DSL quadratic_if; RK4 emitter exact suite",
        tolerance="RK4 golden track",
    ),
    EnrolledEvidence(
        schema_name="theta",
        class_name="ThetaNeuron",
        level="h1_cosim",
        evidence="tests/test_cosimulation.py::_TRANSCENDENTAL_COSIM_MODELS",
        operating_point="schema-DSL theta phase oscillator; Q8.8/Q16 paths",
        tolerance="transcendental tolerance band",
    ),
    EnrolledEvidence(
        schema_name="adex",
        class_name="AdExNeuron",
        level="h1_cosim",
        evidence="tests/test_cosimulation.py::TestQ1616Precision::test_adex_q1616_parity",
        operating_point="schema-DSL adex; Q16.16 RTL vs Python spike count",
        tolerance="class-correct Q16.16 gap bound in test",
    ),
    EnrolledEvidence(
        schema_name="fitzhugh_nagumo",
        class_name="FitzHughNagumoNeuron",
        level="h1_cosim",
        evidence="tests/test_cosimulation.py::TestQ1616Precision::test_fitzhugh_nagumo_q1616_parity",
        operating_point="schema-DSL FHN; edge-crossing; three-way enrolment",
        tolerance="exact or ±1 spike band per test",
    ),
    EnrolledEvidence(
        schema_name="mckean",
        class_name="McKeanNeuron",
        level="h1_cosim",
        evidence="tests/test_cosimulation.py::TestQ1616Precision::test_mckean_q1616_parity",
        operating_point="schema-DSL McKean piecewise-linear oscillator",
        tolerance="three-way exact per enrolment commit",
    ),
    EnrolledEvidence(
        schema_name="glif",
        class_name="GLIFNeuron",
        level="h1_cosim",
        evidence="tests/test_cosimulation.py::TestQ1616Precision::test_glif_q1616_parity",
        operating_point="schema-DSL GLIF four-state RK4 at I=0,15,22,30,45,50 over 1000 steps",
        tolerance="exact hand/schema/Q16.16 spike counts at every operating point",
    ),
    EnrolledEvidence(
        schema_name="izhikevich2007",
        class_name="Izhikevich2007Neuron",
        level="h1_cosim",
        evidence="tests/test_cosimulation.py izhikevich2007 three-way enrolment",
        operating_point="schema-DSL izhikevich2007; Euler; hand/schema/RTL",
        tolerance="exact Q16.16 spike parity per enrolment",
    ),
    EnrolledEvidence(
        schema_name="dpi_neuron",
        class_name="DPINeuron",
        level="h1_cosim",
        evidence="tests/test_cosimulation.py dpi_neuron three-way enrolment",
        operating_point="schema-DSL dpi_neuron (DYNAP-SE class)",
        tolerance="exact three-way spike parity",
    ),
    EnrolledEvidence(
        schema_name="mihalas_niebur",
        class_name="MihalasNieburNeuron",
        level="h1_cosim",
        evidence=(
            "tests/test_cosimulation.py::TestTierBModelCosim::"
            "test_mihalas_niebur_q1616_exact_operating_set; "
            "test_mihalas_niebur_q1616_declares_i3_boundary"
        ),
        operating_point=(
            "schema-DSL mihalas_niebur RK4; hand/schema/Q16.16 RTL; 1000 steps at I=0 through I=6"
        ),
        tolerance=(
            "exact at ten enrolled currents; declared 111/112 RTL boundary at I=3 over 1000 steps"
        ),
    ),
    EnrolledEvidence(
        schema_name="morris_lecar",
        class_name="MorrisLecarNeuron",
        level="h1_cosim",
        evidence="tests/test_cosimulation.py::TestQ1616Precision::test_morris_lecar_q1616_parity",
        operating_point="schema-DSL Morris-Lecar conductance; rk4/crossing",
        tolerance="spike-count parity; not full state trajectory",
    ),
    EnrolledEvidence(
        schema_name="hodgkin_huxley",
        class_name="HodgkinHuxleyNeuron",
        level="h1_cosim",
        evidence="tests/test_cosimulation.py::TestQ1616Precision::test_hodgkin_huxley_q1616_macrostep_parity",
        operating_point="schema-DSL HH macro-step RK4; hand==schema exact",
        tolerance="Q16.16 ±1 bounded window",
    ),
    EnrolledEvidence(
        schema_name="connor_stevens",
        class_name="ConnorStevensNeuron",
        level="h1_cosim",
        evidence="tests/test_cosimulation.py::TestQ1616Precision::test_connor_stevens_q1616_macrostep_parity",
        operating_point="schema-DSL Connor-Stevens macro-step; substeps",
        tolerance="hand==schema exact; Q16.16 bounded band",
    ),
    EnrolledEvidence(
        schema_name="exp_if",
        class_name="ExpIFNeuron",
        level="h0_compile",
        evidence="tests/test_cosimulation.py::TestSchemaGapModelCosim::_SCHEMA_GAP_COMPILE_ONLY",
        operating_point="schema-DSL exp_if; iverilog-valid RTL only",
        tolerance="compile-only (no spike-parity claim)",
    ),
    EnrolledEvidence(
        schema_name="hindmarsh_rose",
        class_name="HindmarshRoseNeuron",
        level="h0_compile",
        evidence="tests/test_cosimulation.py::TestSchemaGapModelCosim::_SCHEMA_GAP_COMPILE_ONLY",
        operating_point="schema-DSL hindmarsh_rose; compile-only",
        tolerance="compile-only (chaotic sensitivity)",
    ),
    EnrolledEvidence(
        schema_name="rulkov_map",
        class_name="RulkovMapNeuron",
        level="h0_compile",
        evidence="tests/test_cosimulation.py::TestSchemaGapModelCosim::_SCHEMA_GAP_COMPILE_ONLY",
        operating_point="schema-DSL rulkov_map; map mode; compile-only for parity class",
        tolerance="compile-only or map-suite separate gates",
    ),
    EnrolledEvidence(
        schema_name="wang_buzsaki",
        class_name="WangBuzsakiNeuron",
        level="h1_cosim",
        evidence="tests/test_cosimulation.py::TestQ1616Precision::test_wang_buzsaki_q1616_parity",
        operating_point="schema-DSL wang_buzsaki (peer lane rewriting integrator)",
        tolerance="existing test; facets deferred",
        skip_apply=True,
        skip_reason="Peer agent (WC-A5) owns Wang-Buzsaki Gauss-Seidel re-enrolment",
    ),
)


@dataclass(frozen=True, slots=True)
class ReadinessRow:
    """Live readiness vs expected evidence for one enrolled model."""

    schema_name: str
    class_name: str
    level: EvidenceLevel
    science: int | None
    silicon: int | None
    has_dynamics: bool
    dynamics_faithful: bool
    compiles: bool
    cosim_validated: bool
    expected_science_min: int
    expected_silicon: int | None
    gap: list[str]
    skip_apply: bool
    skip_reason: str
    evidence: str


def _expected_science_min(entry: EnrolledEvidence, has_dynamics: bool, s0s3: int) -> int:
    """Minimum science tier once facets are honestly wired.

    S4/S5 only open when the S0–S3 curation kernel already reaches 3 (two
    backends + reproducibility anchors). Facet wiring never fabricates that
    kernel — so expected science is capped by ``s0s3`` until curation lands.
    """
    if s0s3 < 3:
        return s0s3
    if entry.level == "h1_cosim" and has_dynamics:
        return 5
    if entry.level == "h0_compile" and has_dynamics:
        # Compile-only cannot claim class-validated spike parity.
        return 4
    return s0s3


def _expected_silicon(entry: EnrolledEvidence) -> int | None:
    if entry.level == "h1_cosim":
        return 1
    if entry.level == "h0_compile":
        return 0
    return None


def _gaps(entry: EnrolledEvidence, row_fields: dict[str, Any]) -> list[str]:
    gaps: list[str] = []
    if entry.skip_apply:
        return gaps
    exp_s = row_fields["expected_science_min"]
    exp_h = row_fields["expected_silicon"]
    science = row_fields["science"]
    silicon = row_fields["silicon"]
    s0s3 = row_fields["s0s3"]
    if science is None:
        gaps.append("descriptor_missing")
        return gaps
    # H1 spike-parity claims require declared dynamics; H0 compile-only does not.
    if entry.level == "h1_cosim" and not row_fields["has_dynamics"]:
        gaps.append("missing_dynamics_facet")
    if (
        entry.level in ("h1_cosim", "h0_compile")
        and row_fields["has_dynamics"]
        and not row_fields["dynamics_faithful"]
    ):
        gaps.append("validation.dynamics_faithful_unset")
    if (
        entry.level == "h1_cosim"
        and row_fields["has_dynamics"]
        and row_fields["validation_metric"] == "none"
    ):
        gaps.append("validation.metric_unset")
    if exp_h is not None:
        if not row_fields["compiles"]:
            gaps.append("silicon.compiles_unset")
        if exp_h >= 1 and not row_fields["cosim_validated"]:
            gaps.append("silicon.cosim_validated_unset")
        if silicon is None or silicon < exp_h:
            gaps.append(f"silicon_below_H{exp_h}(got={silicon})")
    # Science floors only when the curation kernel already permits S4+.
    if s0s3 >= 3 and science is not None and science < exp_s:
        gaps.append(f"science_below_S{exp_s}(got=S{science})")
    return gaps


def build_rows() -> list[ReadinessRow]:
    """Build the live readiness inventory for the enrolled shortlist."""
    rows: list[ReadinessRow] = []
    for entry in ENROLLED:
        desc = load_descriptor(entry.class_name)
        if desc is None:
            rows.append(
                ReadinessRow(
                    schema_name=entry.schema_name,
                    class_name=entry.class_name,
                    level=entry.level,
                    science=None,
                    silicon=None,
                    has_dynamics=False,
                    dynamics_faithful=False,
                    compiles=False,
                    cosim_validated=False,
                    expected_science_min=3,
                    expected_silicon=_expected_silicon(entry),
                    gap=["descriptor_missing"],
                    skip_apply=entry.skip_apply,
                    skip_reason=entry.skip_reason,
                    evidence=entry.evidence,
                )
            )
            continue
        has_dynamics = bool(desc.dynamics)
        s0s3 = descriptor_completeness_tier(desc)
        science = science_tier(desc)
        silicon = silicon_tier(desc)
        dynamics_faithful = desc.validation.dynamics_faithful
        validation_metric = desc.validation.metric
        compiles = desc.silicon.compiles
        cosim_validated = desc.silicon.cosim_validated
        expected_science_min = _expected_science_min(entry, has_dynamics, s0s3)
        expected_silicon = _expected_silicon(entry)
        fields: dict[str, Any] = {
            "science": science,
            "silicon": silicon,
            "s0s3": s0s3,
            "has_dynamics": has_dynamics,
            "dynamics_faithful": dynamics_faithful,
            "validation_metric": validation_metric,
            "compiles": compiles,
            "cosim_validated": cosim_validated,
            "expected_science_min": expected_science_min,
            "expected_silicon": expected_silicon,
        }
        rows.append(
            ReadinessRow(
                schema_name=entry.schema_name,
                class_name=entry.class_name,
                level=entry.level,
                science=science,
                silicon=silicon,
                has_dynamics=has_dynamics,
                dynamics_faithful=dynamics_faithful,
                compiles=compiles,
                cosim_validated=cosim_validated,
                expected_science_min=expected_science_min,
                expected_silicon=expected_silicon,
                gap=_gaps(entry, fields),
                skip_apply=entry.skip_apply,
                skip_reason=entry.skip_reason,
                evidence=entry.evidence,
            )
        )
    return rows


def index_payload(rows: list[ReadinessRow]) -> dict[str, Any]:
    """Return a JSON-serialisable index document."""
    return {
        "schema_version": SCHEMA_VERSION,
        "repo": str(_REPO_ROOT),
        "descriptor_dir": str(DESCRIPTOR_DIR),
        "enrolled_count": len(ENROLLED),
        "rows": [asdict(row) for row in rows],
        "gap_count": sum(1 for row in rows if row.gap and not row.skip_apply),
        "notes": (
            "Facets describe schema-DSL co-simulation evidence. Hand model classes "
            "may differ; see dual-lens audit P3-02."
        ),
    }


def validation_section(entry: EnrolledEvidence, *, has_dynamics: bool) -> dict[str, Any]:
    """Build a Validation facet for an enrolled model.

    ``dynamics_faithful`` is only set when the descriptor already carries a
    dynamics map — never claim S4-ready fidelity without declared equations.
    """
    if entry.level == "h1_cosim" and has_dynamics:
        return {
            "dynamics_faithful": True,
            "metric": "parity",
            "operating_point": entry.operating_point,
            "tolerance": entry.tolerance,
            "evidence": entry.evidence,
        }
    # H0 compile-only, or H1 without a dynamics facet yet: record evidence but
    # do not assert dynamics_faithful.
    return {
        "dynamics_faithful": bool(has_dynamics and entry.level == "h0_compile"),
        "metric": "none",
        "operating_point": entry.operating_point,
        "tolerance": entry.tolerance,
        "evidence": entry.evidence,
    }


def _mapping_has_content(value: object) -> bool:
    """Return True when a dynamics (or similar) mapping has at least one entry."""
    return isinstance(value, dict) and bool(value)


def silicon_section(entry: EnrolledEvidence) -> dict[str, Any]:
    """Build a Silicon facet for an enrolled model."""
    if entry.level == "h1_cosim":
        return {
            "compiles": True,
            "cosim_validated": True,
            "cosim_evidence": entry.evidence,
            "target_tier": "H1",
            "terminal_reason": (
                "Point-neuron schema→RTL path; higher silicon rungs need "
                "synth/timing/formal programmes."
            ),
        }
    return {
        "compiles": True,
        "cosim_validated": False,
        "cosim_evidence": "",
        "target_tier": "H0",
        "terminal_reason": (
            f"Compile-clean RTL only; spike-parity not claimed for this class ({entry.tolerance})."
        ),
    }


def apply_facets(entries: tuple[EnrolledEvidence, ...] | None = None) -> list[str]:
    """Write validation/silicon facets onto enrolled descriptors.

    Returns
    -------
    list[str]
        Human-readable lines describing each write or skip.
    """
    selected = entries if entries is not None else ENROLLED
    lines: list[str] = []
    for entry in selected:
        if entry.skip_apply:
            lines.append(f"SKIP {entry.class_name}: {entry.skip_reason}")
            continue
        path = descriptor_path(entry.class_name)
        if not path.is_file():
            lines.append(f"MISS {entry.class_name}: no descriptor at {path}")
            continue
        payload = load_descriptor_payload(entry.class_name)
        if payload is None:
            lines.append(f"MISS {entry.class_name}: payload unreadable")
            continue
        payload = dict(payload)
        has_dynamics = bool(_mapping_has_content(payload.get("dynamics")))
        current = load_descriptor(entry.class_name)
        if current is None:
            lines.append(f"MISS {entry.class_name}: descriptor reload failed")
            continue
        s0s3 = descriptor_completeness_tier(current)
        current_science = science_tier(current)
        current_silicon = silicon_tier(current)
        current_fields: dict[str, Any] = {
            "science": current_science,
            "silicon": current_silicon,
            "s0s3": s0s3,
            "has_dynamics": has_dynamics,
            "dynamics_faithful": current.validation.dynamics_faithful,
            "validation_metric": current.validation.metric,
            "compiles": current.silicon.compiles,
            "cosim_validated": current.silicon.cosim_validated,
            "expected_science_min": _expected_science_min(entry, has_dynamics, s0s3),
            "expected_silicon": _expected_silicon(entry),
        }
        if not _gaps(entry, current_fields):
            lines.append(
                f"PRESERVED {entry.class_name}: existing S{current_science} "
                f"H{current_silicon} meets or exceeds {entry.level}"
            )
            continue
        payload["validation"] = validation_section(entry, has_dynamics=has_dynamics)
        payload["silicon"] = silicon_section(entry)
        rendered = _DESCRIPTOR_HEADER + tomli_w.dumps(payload)
        path.write_text(rendered, encoding="utf-8")
        # Re-load and score for the log line.
        desc = load_descriptor(entry.class_name)
        if desc is None:
            lines.append(f"WROTE {entry.class_name} but reload failed")
            continue
        lines.append(
            f"APPLIED {entry.class_name}: S{science_tier(desc)} "
            f"H{silicon_tier(desc)} ← {entry.level} ({entry.schema_name})"
        )
    return lines


def report_text(rows: list[ReadinessRow]) -> str:
    """Format a human-readable table."""
    lines = [
        f"Readiness evidence index ({SCHEMA_VERSION})",
        f"{'class':28} {'schema':18} {'S':3} {'H':5} {'expS':4} {'expH':4} gaps",
        "-" * 100,
    ]
    for row in rows:
        h = "none" if row.silicon is None else str(row.silicon)
        s = "—" if row.science is None else str(row.science)
        exp_h = "—" if row.expected_silicon is None else str(row.expected_silicon)
        gap = ",".join(row.gap) if row.gap else ("skip" if row.skip_apply else "ok")
        lines.append(
            f"{row.class_name:28} {row.schema_name:18} {s:3} {h:5} "
            f"{row.expected_science_min:<4} {exp_h:4} {gap}"
        )
    open_gaps = sum(1 for row in rows if row.gap and not row.skip_apply)
    lines.append("-" * 100)
    lines.append(f"open_gaps={open_gaps} enrolled={len(rows)}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", help="print the inventory table")
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 when any non-skipped enrolled model still has readiness gaps",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write validation/silicon facets onto enrolled descriptors (skip peer-owned)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="write the full index JSON to this path",
    )
    args = parser.parse_args(argv)

    if not any((args.report, args.check, args.apply, args.json is not None)):
        args.report = True

    if args.apply:
        for line in apply_facets():
            print(line)

    rows = build_rows()
    if args.report or args.check:
        print(report_text(rows), end="")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(index_payload(rows), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.json}")

    if args.check:
        open_gaps = [row for row in rows if row.gap and not row.skip_apply]
        if open_gaps:
            print(f"check FAILED: {len(open_gaps)} enrolled model(s) with readiness gaps")
            return 1
        print("check PASSED: enrolled shortlist facets match expected evidence floors")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

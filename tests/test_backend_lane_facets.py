# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The readiness facets of the backend lanes: subjects, artefacts, evidence field, build inputs

"""Bind what the readiness facet of a backend lane requires.

Every lane a descriptor may declare has one facet. A lane's evidence is the
text its descriptor table states. What must still be the same for a receipt to
hold is what a repository can recompute on any host: the descriptor contract,
the model module, the validator and the lane's sources. The library a run
loaded is a fact of that run, kept as an artefact of the receipt, never as a
subject: a built file is absent on a clean checkout and differs between hosts.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from sc_neurocore.neurons.facet_receipts import (
    FACET_BY_NAME,
    FACETS,
    INVALIDATION_MATRIX,
    FacetReceipt,
    Subject,
    backend_lane,
    credit_problems,
    facets_invalidated_by,
)
from sc_neurocore.neurons.model_descriptor import BACKEND_LANES, parse_model_descriptor
from sc_neurocore.neurons.receipt_execution import LANE_PROPERTY, NATIVE_BINARY_PROPERTY
from sc_neurocore.neurons.readiness import (
    LANE_BUILD_INPUTS,
    declared_facets,
    facet_evidence_field,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DIGEST = "0" * 64
# The lanes whose model code is not the Python module, and those of them that build a library.
_NATIVE_LANES = ("rust", "rust_safety", "julia", "go", "mojo")
_BUILT_LANES = ("rust", "rust_safety", "go", "mojo")
_SOFTWARE_SUBJECTS = ("descriptor-contract", "model-module", "validator")


def _descriptor(backends: dict[str, Any]) -> Any:
    """Return a parsed minimal descriptor with the given backend section."""
    return parse_model_descriptor(
        {
            "metadata": {
                "schema_version": 2,
                "name": "Leaky demonstration",
                "class_name": "LapicqueNeuron",
                "module": "lapicque",
            },
            "state": {"v": {"init": 0.0}},
            "parameters": {"tau": {"default": 20.0}},
            "integration": {"dt": 0.1},
            "backends": backends,
        }
    )


def _receipt(facet: str, artefacts: tuple[Subject, ...] = ()) -> FacetReceipt:
    """Return a sealed, passed receipt of one facet with every required subject.

    A backend facet's report names its lane and, per ``native-binary`` artefact,
    the library the run loaded, as a run under the lane fixture writes them.
    """
    spec = FACET_BY_NAME[facet]
    lane = backend_lane(facet)
    properties = [(LANE_PROPERTY, lane)] if lane else []
    properties += [
        (NATIVE_BINARY_PROPERTY, f"{artefact.path}={artefact.sha256}")
        for artefact in artefacts
        if lane and artefact.kind == "native-binary"
    ]
    block = "".join(f'<property name="{key}" value="{value}" />' for key, value in properties)
    case = '<testcase classname="tests.test_lapicque_backend_parity" name="test_x">'
    case += f"<properties>{block}</properties></testcase>" if block else "</testcase>"
    return FacetReceipt(
        class_name="LapicqueNeuron",
        facet=facet,
        profile="lapicque",
        claim_scope=spec.claim_scope,
        subjects=tuple(
            Subject(kind, f"path/{kind}.txt", _DIGEST) for kind in spec.required_subjects
        ),
        evidence_refs=("tests/test_lapicque_backend_parity.py::test_x",),
        command=("python", "-m", "pytest", "tests/test_lapicque_backend_parity.py::test_x"),
        tool={"name": "pytest", "version": "9.2.0"},
        extra_tools={},
        runtime={"python": "3.12"},
        validator={
            "name": "tools/facet_receipt.py",
            "execution_contract": "pytest-junit.v1",
            "junit_xml": f"<testsuite>{case}</testsuite>",
        },
        outcome="passed",
        exit_code=0,
        counts={"collected": 1, "passed": 1, "failed": 0, "errors": 0, "skipped": 0},
        recorded_at="2026-11-12T12:00:00Z",
        artifacts=artefacts,
    ).sealed()


def test_every_lane_has_one_facet_in_the_order_of_the_lanes() -> None:
    """The backend facets are the six lanes, and the table has sixteen facets."""
    backend = [spec for spec in FACETS if spec.name.startswith("backend:")]
    assert [spec.name for spec in backend] == [f"backend:{lane}" for lane in BACKEND_LANES]
    assert all(spec.axis == "software" and spec.rung is None for spec in backend)
    assert all(spec.claim_scope == "" for spec in backend)
    assert len(FACETS) == 16
    assert len(FACET_BY_NAME) == 16


@pytest.mark.parametrize("lane", BACKEND_LANES)
def test_a_lane_facet_reads_its_evidence_from_the_lane_table(lane: str) -> None:
    """The evidence field of a lane's facet is that lane's entry in the descriptor.

    Parameters
    ----------
    lane : str
        One of the six lanes.
    """
    assert FACET_BY_NAME[f"backend:{lane}"].evidence_field == f"backends.{lane}.evidence"


def test_the_reference_lane_requires_no_native_source() -> None:
    """The Python lane's code is the model module itself."""
    spec = FACET_BY_NAME["backend:python"]
    assert spec.required_subjects == _SOFTWARE_SUBJECTS
    assert spec.optional_subjects == ("schema-profile", "source-reference")
    assert spec.required_artifacts == ()


@pytest.mark.parametrize("lane", _NATIVE_LANES)
def test_a_native_lane_requires_its_sources_and_never_its_binary_as_a_subject(lane: str) -> None:
    """Freshness is carried by sources; a built library is not a subject of any lane.

    Parameters
    ----------
    lane : str
        A lane whose model code is not the Python module.
    """
    spec = FACET_BY_NAME[f"backend:{lane}"]
    assert spec.required_subjects == (*_SOFTWARE_SUBJECTS, "native-backend")
    assert spec.optional_subjects == ("schema-profile", "source-reference")
    assert "native-binary" not in spec.subjects
    assert INVALIDATION_MATRIX[spec.name] == spec.subjects


@pytest.mark.parametrize("lane", BACKEND_LANES)
def test_only_a_lane_that_builds_a_library_requires_it_as_an_artefact(lane: str) -> None:
    """Rust, the safety crate, Go and Mojo load a built library; Julia and Python do not.

    Parameters
    ----------
    lane : str
        One of the six lanes.
    """
    expected = ("native-binary",) if lane in _BUILT_LANES else ()
    assert FACET_BY_NAME[f"backend:{lane}"].required_artifacts == expected


def test_no_facet_outside_the_lanes_requires_an_artefact() -> None:
    """The science and silicon facets keep their inputs as subjects only."""
    others = [spec for spec in FACETS if not spec.name.startswith("backend:")]
    assert len(others) == 10
    assert all(spec.required_artifacts == () for spec in others)


def test_a_change_of_a_lane_source_touches_the_native_lanes_and_a_binary_touches_none() -> None:
    """The invalidation matrix follows the subjects: sources invalidate, a binary does not."""
    assert set(facets_invalidated_by("native-backend")) == {
        f"backend:{lane}" for lane in _NATIVE_LANES
    }
    assert facets_invalidated_by("native-binary") == ()


def test_the_evidence_of_a_lane_is_what_its_table_states() -> None:
    """A lane with evidence gives it; a lane without, and a lane that is absent, give nothing."""
    descriptor = _descriptor(
        {
            "python": {
                "status": "implemented",
                "evidence": "tests/test_lapicque_reference.py::test_golden_trace",
            },
            "rust": {"status": "implemented", "parity": "ulp-bounded"},
            "rust_safety": {
                "status": "implemented",
                "evidence": "tests/test_lapicque_backend_parity.py; tests/test_lapicque_abi.py",
            },
        }
    )
    assert facet_evidence_field(descriptor, FACET_BY_NAME["backend:python"]) == (
        "tests/test_lapicque_reference.py::test_golden_trace"
    )
    assert facet_evidence_field(descriptor, FACET_BY_NAME["backend:rust"]) == ""
    assert facet_evidence_field(descriptor, FACET_BY_NAME["backend:rust_safety"]) == (
        "tests/test_lapicque_backend_parity.py; tests/test_lapicque_abi.py"
    )
    assert facet_evidence_field(descriptor, FACET_BY_NAME["backend:go"]) == ""
    assert facet_evidence_field(descriptor, FACET_BY_NAME["cosim"]) == ""


def test_a_lane_is_declared_when_its_own_table_says_implemented() -> None:
    """Each of the six lanes is declared by its table; a planned lane is not declared."""
    declared = declared_facets(
        _descriptor(
            {
                "python": {"status": "implemented"},
                "rust_safety": {"status": "implemented"},
                "go": {"status": "planned"},
                "mojo": {"status": "unsupported"},
            }
        )
    )
    assert {name: value for name, value in declared.items() if name.startswith("backend:")} == {
        "backend:python": True,
        "backend:rust": False,
        "backend:rust_safety": True,
        "backend:julia": False,
        "backend:go": False,
        "backend:mojo": False,
    }


def test_a_table_under_the_legacy_name_declares_no_facet() -> None:
    """The earlier name of the Rust lane is read by the parser and seen by no facet."""
    declared = declared_facets(
        _descriptor({"python": {"status": "implemented"}, "rust_engine": {"status": "implemented"}})
    )
    assert declared["backend:rust"] is False
    assert sorted(name for name in declared if name.startswith("backend:")) == sorted(
        f"backend:{lane}" for lane in BACKEND_LANES
    )


@pytest.mark.parametrize("lane", _BUILT_LANES)
def test_a_receipt_of_a_built_lane_must_carry_the_library_it_loaded(lane: str) -> None:
    """Without the artefact the receipt cannot credit; with it, nothing is missing.

    Parameters
    ----------
    lane : str
        A lane that loads a built library.
    """
    facet = f"backend:{lane}"
    assert credit_problems(_receipt(facet)) == (
        "required artefact kind 'native-binary' is missing",
        f"test case tests.test_lapicque_backend_parity.test_x of lane {lane} names no library",
    )
    library = Subject("native-binary", f"lib{lane}.so", _DIGEST)
    assert credit_problems(_receipt(facet, (library,))) == ()


@pytest.mark.parametrize("facet", ["backend:python", "backend:julia", "cosim", "class_validated"])
def test_a_facet_without_a_required_artefact_takes_none(facet: str) -> None:
    """An artefact of a kind the facet does not require is not an output of that facet.

    Parameters
    ----------
    facet : str
        A facet whose receipts carry no artefact.
    """
    assert credit_problems(_receipt(facet)) == ()
    library = Subject("native-binary", "libx.so", _DIGEST)
    assert credit_problems(_receipt(facet, (library,))) == (
        f"artefact kind 'native-binary' is not an output of {facet}",
    )


def test_an_artefact_of_another_kind_does_not_stand_for_the_library() -> None:
    """A report in the artefacts neither satisfies the requirement nor is accepted."""
    report = Subject("report", "build/report.txt", _DIGEST)
    assert credit_problems(_receipt("backend:go", (report,))) == (
        "required artefact kind 'native-binary' is missing",
        "artefact kind 'report' is not an output of backend:go",
        "test case tests.test_lapicque_backend_parity.test_x of lane go names no library",
        "receipt artefacts are not the libraries its report names",
    )


def test_an_input_kind_of_a_facet_is_no_output_of_it() -> None:
    """A report is a subject of the synthesis facet; among the artefacts it is refused."""
    assert credit_problems(_receipt("synthesis")) == ()
    report = Subject("report", "build/report.txt", _DIGEST)
    assert credit_problems(_receipt("synthesis", (report,))) == (
        "artefact kind 'report' is not an output of synthesis",
    )


def test_the_build_inputs_of_each_lane_are_its_tracked_manifests_locks_and_configurations() -> None:
    """Every lane names the files that fix its build; each is a tracked file of the repository.

    A build reads the Cargo configurations from the directory it starts in upwards.
    The engine wheel, the library the Rust lane loads, is built from ``bridge/``
    (its ``pyproject.toml`` names the manifest and the module), so it reads the
    bridge's and the root configuration; the engine's own builds, benchmarks and
    fuzzing, start in ``engine/`` and read the engine's. The build tool's pin is an
    input as the toolchain's is; the record of the vendored crate's origin is read
    by no build and is listed because the wheel's cache key lists it. The safety
    crate is built in its own folder and from the root with ``--manifest-path``;
    both builds read the root configuration.
    """
    assert LANE_BUILD_INPUTS == {
        "python": (),
        "rust": (
            ".cargo/config.toml",
            "Cargo.lock",
            "Cargo.toml",
            "bridge/.cargo/config.toml",
            "bridge/pyproject.toml",
            "bridge/rayon_source_patch.toml",
            "engine/.cargo/config.toml",
            "engine/Cargo.toml",
            "engine/rust-toolchain.toml",
            "requirements/maturin.txt",
        ),
        "rust_safety": (
            ".cargo/config.toml",
            "src/sc_neurocore/accel/rust/Cargo.lock",
            "src/sc_neurocore/accel/rust/Cargo.toml",
        ),
        "julia": ("src/sc_neurocore/juliapkg.json",),
        "go": ("src/sc_neurocore/accel/go/go.mod",),
        "mojo": ("src/sc_neurocore/accel/mojo/pixi.lock", "src/sc_neurocore/accel/mojo/pixi.toml"),
    }
    assert tuple(LANE_BUILD_INPUTS) == BACKEND_LANES
    listed = sorted({path for paths in LANE_BUILD_INPUTS.values() for path in paths})
    assert all((_REPO_ROOT / path).is_file() for path in listed)
    if (_REPO_ROOT / ".git").exists():
        tracked = subprocess.run(
            ["git", "ls-files", "--", *listed],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.split()
        assert sorted(tracked) == listed

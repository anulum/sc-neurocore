# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — A backend lane's receipt reads the lane that ran

"""Bind what the receipt of a backend lane must show of the run it records.

A backend lane is credited only from a receipt whose retained report shows, case
by case, that the facet's lane ran, and whose artefacts are exactly the native
libraries that report names. Its claim scope is the word the model's descriptor
gives the lane today: the parity word of a compiled lane, ``reference`` for the
Python lane. The recorder writes the three from the run and the descriptor; the
verifier derives them again.
"""

from __future__ import annotations

import importlib.util
import os
import stat
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any
from xml.sax.saxutils import quoteattr

import pytest
import tomli_w

from sc_neurocore.accel import lapicque as backends
from sc_neurocore.neurons import model_catalogue
from sc_neurocore.neurons.evidence_references import sha256_file
from sc_neurocore.neurons.facet_receipts import (
    FacetReceipt,
    Subject,
    backend_lane,
    credit_problems,
    lane_run_evidence,
)
from sc_neurocore.neurons.lane_runs import LaneRun
from sc_neurocore.neurons.model_descriptor import BACKEND_LANES, parse_model_descriptor
from sc_neurocore.neurons.models.lapicque import LapicqueNeuron
from sc_neurocore.neurons.readiness import (
    LANE_BUILD_INPUTS,
    backend_claim_scope,
    derive_subjects,
    verify_receipt,
)
from sc_neurocore.neurons.receipt_execution import (
    LANE_PROPERTY,
    NATIVE_BINARY_PROPERTY,
    junit_checks,
)

pytest_plugins = ("tests.lane_evidence_support",)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DESCRIPTOR = "src/sc_neurocore/neurons/model_descriptors/LapicqueNeuron.toml"
_MODEL = "src/sc_neurocore/neurons/models/lapicque.py"
_VALIDATOR = "tests/test_lane_probe.py"
_NODE = f"{_VALIDATOR}::test_lane"
_CASE = "tests.test_lane_probe.test_lane"
_SOURCE = "engine/src/neuron/lapicque.rs"
_LIBRARY = "liblapicque.so"
_A = "a" * 64
_B = "b" * 64
# The lanes whose run loads a library built from the lane's sources.
_BUILDS = ("rust", "rust_safety", "go", "mojo")
_THIS = "tests/test_backend_lane_receipts.py"
_REFERENCE_RUN = f"{_THIS}::test_the_reference_lane_writes_its_run_into_the_report"
_RUST_RUN = f"{_THIS}::test_the_rust_lane_writes_its_run_into_the_report"


def _payload(lane: str, parity: str) -> dict[str, Any]:
    """Return the shipped Lapicque descriptor whose ``lane`` names the probe as its evidence."""
    payload = tomllib.loads((_REPO_ROOT / _DESCRIPTOR).read_text(encoding="utf-8"))
    table: dict[str, Any] = {"status": "implemented", "parity": parity, "evidence": _NODE}
    if lane != "python":
        table["sources"] = [_SOURCE]
    payload["backends"][lane] = table
    return payload


def _root(tmp_path: Path, lane: str, parity: str = "exact") -> Path:
    """Write a repository root holding what a receipt of ``lane`` binds."""
    files = {
        _DESCRIPTOR: tomli_w.dumps(_payload(lane, parity)),
        _MODEL: "# the model module\n",
        _VALIDATOR: "def test_lane():\n    pass\n",
        _SOURCE: "// the lane's source\n",
        **{path: f"# {path}\n" for path in LANE_BUILD_INPUTS.get(lane, ())},
    }
    for relative, text in files.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return tmp_path


def _case(
    lane: str | None = "rust",
    libraries: tuple[tuple[str, str], ...] = ((_LIBRARY, _A),),
    *,
    name: str = "test_lane",
    skipped: bool = False,
) -> str:
    """Return one test case of the probe validator with its lane evidence."""
    properties = [(LANE_PROPERTY, lane)] if lane is not None else []
    properties += [(NATIVE_BINARY_PROPERTY, f"{file}={digest}") for file, digest in libraries]
    inner = "".join(
        f"<property name={quoteattr(key)} value={quoteattr(value)} />" for key, value in properties
    )
    block = f"<properties>{inner}</properties>" if properties else ""
    skip = '<skipped message="skip" />' if skipped else ""
    return f'<testcase classname="tests.test_lane_probe" name="{name}">{block}{skip}</testcase>'


def _report(*cases: str) -> str:
    """Return a JUnit report holding the given test cases."""
    return f'<testsuite name="pytest">{"".join(cases)}</testsuite>'


def _libraries(lane: str) -> tuple[tuple[str, str], ...]:
    """Return the libraries a run of ``lane`` reports in these cases."""
    return ((_LIBRARY, _A),) if lane in _BUILDS else ()


def _receipt(
    root: Path,
    lane: str,
    *,
    xml: str | None = None,
    artifacts: tuple[Subject, ...] | None = None,
    claim_scope: str | None = None,
) -> FacetReceipt:
    """Return a sealed receipt of ``backend:<lane>`` for the probe run, as the recorder writes it."""
    facet = f"backend:{lane}"
    report = xml if xml is not None else _report(_case(lane, _libraries(lane)))
    carried = (
        artifacts
        if artifacts is not None
        else tuple(Subject("native-binary", file, digest) for file, digest in _libraries(lane))
    )
    return FacetReceipt(
        class_name="LapicqueNeuron",
        facet=facet,
        profile="lapicque",
        claim_scope=claim_scope
        if claim_scope is not None
        else ("reference" if lane == "python" else "exact"),
        subjects=derive_subjects("LapicqueNeuron", facet, repo_root=root, evidence_refs=[_NODE]),
        evidence_refs=(_NODE,),
        command=("python", "-m", "pytest", _NODE),
        tool={"name": "pytest", "version": "9.2.0"},
        extra_tools={},
        runtime={"python": "3.12"},
        validator={
            "name": "tools/facet_receipt.py",
            "execution_contract": "pytest-junit.v1",
            "junit_xml": report,
        },
        outcome="passed",
        exit_code=0,
        counts=junit_checks(report)[0],
        recorded_at="2026-11-12T17:00:00Z",
        artifacts=carried,
    ).sealed()


def _verdict(root: Path, receipt: FacetReceipt) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Return the verifier's status, changed subjects and problems at ``root``."""
    return verify_receipt(receipt, class_name="LapicqueNeuron", repo_root=root)


@pytest.mark.parametrize("lane", BACKEND_LANES)
def test_a_receipt_of_the_lane_that_ran_is_bound(tmp_path: Path, lane: str) -> None:
    """Every lane: the report names the lane, the artefacts are its libraries, the word is today's.

    Parameters
    ----------
    lane : str
        The lane of the facet; the compiled lanes report one library, Julia and
        Python none.
    """
    root = _root(tmp_path, lane)
    assert _verdict(root, _receipt(root, lane)) == ("bound", (), ())


def test_a_case_that_ran_another_lane_cannot_credit(tmp_path: Path) -> None:
    """A fallback to the reference lane under a Rust receipt is named, case by case."""
    root = _root(tmp_path, "rust")
    receipt = _receipt(root, "rust", xml=_report(_case("python", ())), artifacts=())
    status, changed, problems = _verdict(root, receipt)
    assert (status, changed) == ("invalid", ())
    assert f"test case {_CASE} ran lane python, not rust" in problems
    assert f"test case {_CASE} of lane rust names no library" not in problems


def test_a_case_that_names_no_lane_cannot_credit(tmp_path: Path) -> None:
    """A case without the lane property is no evidence, and its library is no artefact."""
    root = _root(tmp_path, "go")
    receipt = _receipt(root, "go", xml=_report(_case(None)))
    status, _changed, problems = _verdict(root, receipt)
    assert status == "invalid"
    assert f"test case {_CASE} reports no lane" in problems
    assert "receipt artefacts are not the libraries its report names" in problems


def test_a_receipt_without_the_library_its_run_loaded_cannot_credit(tmp_path: Path) -> None:
    """The library a compiled lane loaded must be among the receipt's artefacts."""
    root = _root(tmp_path, "mojo")
    status, _changed, problems = _verdict(root, _receipt(root, "mojo", artifacts=()))
    assert status == "invalid"
    assert "required artefact kind 'native-binary' is missing" in problems
    assert "receipt artefacts are not the libraries its report names" in problems


@pytest.mark.parametrize(
    "artifact",
    [
        Subject("native-binary", _LIBRARY, _B),
        Subject("native-binary", "libother.so", _A),
    ],
)
def test_an_artefact_that_is_not_the_reported_library_cannot_credit(
    tmp_path: Path, artifact: Subject
) -> None:
    """Another digest or another file than the report's is refused.

    Parameters
    ----------
    artifact : Subject
        The receipt's one artefact, differing from the report in digest or name.
    """
    root = _root(tmp_path, "rust")
    status, _changed, problems = _verdict(root, _receipt(root, "rust", artifacts=(artifact,)))
    assert status == "invalid"
    assert problems == ("receipt artefacts are not the libraries its report names",)


@pytest.mark.parametrize("lane", _BUILDS)
def test_a_case_of_a_compiled_lane_that_loaded_no_library_cannot_credit(
    tmp_path: Path, lane: str
) -> None:
    """A compiled lane's case must name the library it ran.

    Parameters
    ----------
    lane : str
        A lane whose run loads a library.
    """
    root = _root(tmp_path, lane)
    receipt = _receipt(root, lane, xml=_report(_case(lane, ())), artifacts=())
    status, _changed, problems = _verdict(root, receipt)
    assert status == "invalid"
    assert f"test case {_CASE} of lane {lane} names no library" in problems


@pytest.mark.parametrize(
    "name", ["sub/liblapicque.so", "sub\\liblapicque.so", ".", "..", "lib\nlapicque.so"]
)
def test_a_library_that_is_not_a_plain_file_name_is_no_artefact(name: str) -> None:
    """A path, a backslash, a dot name or a control character is refused, never recorded.

    Parameters
    ----------
    name : str
        The file name a crafted report gives the library.
    """
    report = _report(_case("rust", ((name, _A),)))
    assert lane_run_evidence("backend:rust", report) == (
        (),
        (f"test case {_CASE} names a library that is not a file name: {name!r}",),
    )


def test_each_library_is_one_artefact_in_name_order() -> None:
    """Two cases that load the same library give it once; two libraries are sorted by name."""
    report = _report(
        _case("rust", (("libb.so", _A), ("liba.so", _B)), name="test_one"),
        _case("rust", (("liba.so", _B),), name="test_two"),
    )
    assert lane_run_evidence("backend:rust", report) == (
        (Subject("native-binary", "liba.so", _B), Subject("native-binary", "libb.so", _A)),
        (),
    )


def test_a_report_whose_every_case_was_skipped_names_no_lane() -> None:
    """With no executed case there is no lane evidence at all."""
    report = _report(_case("go", skipped=True))
    assert lane_run_evidence("backend:go", report) == (
        (),
        ("no executed test case reports a lane",),
    )


def test_a_report_the_lane_reader_refuses_is_named() -> None:
    """A report with a declared DTD gives no evidence and one problem."""
    report = '<!DOCTYPE testsuite [<!ENTITY x "y">]><testsuite name="pytest" />'
    assert lane_run_evidence("backend:rust", report) == (
        (),
        ("lane evidence cannot be read: invalid JUnit report",),
    )


def test_a_facet_of_another_axis_has_no_lane() -> None:
    """Only the backend facets read lane evidence; the others are unchanged."""
    report = _report(_case("python", ()))
    for facet in ("dynamics_faithful", "class_validated", "cosim", "rtl_compile"):
        assert backend_lane(facet) == ""
        assert lane_run_evidence(facet, report) == ((), ())
    assert [backend_lane(f"backend:{lane}") for lane in BACKEND_LANES] == list(BACKEND_LANES)


def test_a_claim_scope_other_than_today_s_word_is_invalid(tmp_path: Path) -> None:
    """A parity word changed after the recording makes the receipt invalid, not stale.

    The descriptor's contract digest does not cover ``[backends]``, so only the
    claim scope can carry the change.
    """
    root = _root(tmp_path, "go", parity="exact")
    receipt = _receipt(root, "go")
    assert _verdict(root, receipt) == ("bound", (), ())
    (root / _DESCRIPTOR).write_text(tomli_w.dumps(_payload("go", "ulp-bounded")), encoding="utf-8")
    assert _verdict(root, receipt) == (
        "invalid",
        (),
        ("claim scope 'exact' is not 'ulp-bounded', the word the descriptor gives lane go today",),
    )


def test_a_lane_whose_parity_word_is_withdrawn_cannot_credit(tmp_path: Path) -> None:
    """A lane declared ``n/a`` has no word a receipt could state."""
    root = _root(tmp_path, "mojo", parity="n/a")
    assert _verdict(root, _receipt(root, "mojo")) == (
        "invalid",
        (),
        ("the descriptor gives lane mojo no parity word",),
    )


def test_the_reference_lane_s_word_is_reference_whatever_its_parity() -> None:
    """Python is the reference; a compiled lane's word is its parity, or none."""
    shipped = parse_model_descriptor(
        tomllib.loads((_REPO_ROOT / _DESCRIPTOR).read_text(encoding="utf-8"))
    )
    assert [backend_claim_scope(shipped, lane) for lane in BACKEND_LANES] == [
        "reference",
        "exact",
        "",
        "ulp-bounded",
        "ulp-bounded",
        "ulp-bounded",
    ]
    for word in ("exact", "ulp-bounded", "approximate"):
        assert backend_claim_scope(parse_model_descriptor(_payload("go", word)), "go") == word
    assert backend_claim_scope(parse_model_descriptor(_payload("go", "n/a")), "go") == ""


def test_an_edited_source_makes_the_receipt_stale_and_names_it(tmp_path: Path) -> None:
    """Freshness is carried by the lane's sources, not by the library."""
    root = _root(tmp_path, "rust")
    receipt = _receipt(root, "rust")
    (root / _SOURCE).write_text("// edited\n", encoding="utf-8")
    assert _verdict(root, receipt) == ("stale", (f"native-backend:{_SOURCE}",), ())


def test_a_receipt_whose_run_skipped_a_case_cannot_credit(tmp_path: Path) -> None:
    """A skipped case carries no lane evidence and still keeps the receipt from credit."""
    root = _root(tmp_path, "rust")
    report = _report(_case(), _case(name="test_other", skipped=True))
    status, _changed, problems = _verdict(root, _receipt(root, "rust", xml=report))
    assert status == "invalid"
    assert problems == ("1 skipped check(s) recorded",)


def test_the_reference_lane_writes_its_run_into_the_report(
    lane_evidence: Callable[[object], LaneRun],
) -> None:
    """A run of the Python lane reports that lane and no library.

    Parameters
    ----------
    lane_evidence : callable
        Writes the model's lane record into this test's report.
    """
    neuron = LapicqueNeuron()
    neuron.simulate(20, 2.2, backend="python")
    assert lane_evidence(neuron) == LaneRun("python", ())


def test_the_rust_lane_writes_its_run_into_the_report(
    lane_evidence: Callable[[object], LaneRun],
) -> None:
    """A run of the Rust lane reports that lane and the engine module it loaded.

    Parameters
    ----------
    lane_evidence : callable
        Writes the model's lane record into this test's report.
    """
    neuron = LapicqueNeuron()
    neuron.simulate(20, 2.2, backend="rust")
    assert lane_evidence(neuron) == LaneRun("rust", backends.lane_libraries("rust"))


def _load_recorder() -> ModuleType:
    """Load the recorder tool as the receipt tests load it."""
    tool_path = _REPO_ROOT / "tools" / "facet_receipt.py"
    spec = importlib.util.spec_from_file_location("facet_receipt_tool", tool_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _recorder_for(
    monkeypatch: pytest.MonkeyPatch, facet: str, node: str, parity: str = "exact"
) -> ModuleType:
    """Load the recorder with a descriptor that names ``node`` as the evidence of the facet's lane.

    No shipped descriptor names a lane's evidence yet, so the descriptor the
    recorder reads is the shipped one with that entry and ``parity`` added.
    """
    payload = tomllib.loads((_REPO_ROOT / _DESCRIPTOR).read_text(encoding="utf-8"))
    payload["backends"][backend_lane(facet)] = {
        "status": "implemented",
        "parity": parity,
        "evidence": node,
    }
    descriptor = parse_model_descriptor(payload)
    monkeypatch.setattr(model_catalogue, "load_descriptor", lambda _name: descriptor)
    return _load_recorder()


def _record(
    recorder: ModuleType,
    tmp_path: Path,
    facet: str,
    node: str,
    *,
    claim_scope: str | None = None,
) -> FacetReceipt:
    """Record ``facet`` from a real run of ``node`` and return the sealed receipt."""
    _path, receipt = recorder.record_receipt(
        model="LapicqueNeuron",
        facet=facet,
        profile="lapicque",
        evidence=[node],
        command=[sys.executable, "-m", "pytest", node, "-q", "-p", "no:nengo"],
        claim_scope=claim_scope,
        receipt_dir=tmp_path / "receipts",
        raw_evidence_dir=tmp_path / "raw",
        recorded_at="2026-11-12T17:00:00Z",
        timeout=300,
    )
    # The recorder is loaded from its file, so its return type is not seen by the type checker.
    assert isinstance(receipt, FacetReceipt)
    return receipt


def _stand_in_tools(directory: Path, *, failing: tuple[str, ...] = ()) -> Path:
    """Write stand-in lane tools that print their name, arguments and working folder."""
    directory.mkdir()
    for name in ("rustc", "cargo", "go", "mojo", "julia"):
        tool = directory / name
        # Shell expansions only: the tests narrow the path to this folder.
        body = (
            "exit 3" if name in failing else 'folder=$(pwd -P)\necho "${0##*/} $* in ${folder##*/}"'
        )
        tool.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
        tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
    return directory


def test_the_recorder_writes_the_reference_lane_s_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A real Python-lane run is recorded under the word ``reference``, with no artefact."""
    recorder = _recorder_for(monkeypatch, "backend:python", _REFERENCE_RUN)
    receipt = _record(recorder, tmp_path, "backend:python", _REFERENCE_RUN)
    assert (receipt.outcome, receipt.claim_scope, receipt.artifacts) == ("passed", "reference", ())
    assert receipt.extra_tools == {}
    assert credit_problems(receipt, class_name="LapicqueNeuron") == ()


def test_the_recorder_writes_the_library_the_rust_lane_loaded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A real Rust-lane run is recorded with the engine module's digest and the toolchain."""
    stand_ins = _stand_in_tools(tmp_path / "bin")
    monkeypatch.setenv("PATH", f"{stand_ins}{os.pathsep}{os.environ.get('PATH', '')}")
    recorder = _recorder_for(monkeypatch, "backend:rust", _RUST_RUN)
    receipt = _record(recorder, tmp_path, "backend:rust", _RUST_RUN)
    (library,) = backends.lane_libraries("rust")
    assert (receipt.outcome, receipt.claim_scope) == ("passed", "exact")
    assert receipt.artifacts == (Subject("native-binary", library.name, sha256_file(library)),)
    assert receipt.extra_tools == {"rustc": "rustc -V in bridge", "cargo": "cargo -V in bridge"}
    assert credit_problems(receipt, class_name="LapicqueNeuron") == ()


def test_the_recorder_marks_a_run_of_another_lane_as_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A Python-lane run recorded for the Rust facet is an error naming the lane that ran."""
    recorder = _recorder_for(monkeypatch, "backend:rust", _REFERENCE_RUN)
    receipt = _record(recorder, tmp_path, "backend:rust", _REFERENCE_RUN)
    message = (
        "test case tests.test_backend_lane_receipts."
        "test_the_reference_lane_writes_its_run_into_the_report ran lane python, not rust"
    )
    assert receipt.outcome == "error"
    assert message in receipt.notes.splitlines()
    assert receipt.artifacts == ()
    assert message in credit_problems(receipt, class_name="LapicqueNeuron")


def test_the_recorder_refuses_a_lane_without_a_parity_word(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A compiled lane declared ``n/a`` cannot be recorded until its word is declared."""
    recorder = _recorder_for(monkeypatch, "backend:go", _RUST_RUN, parity="n/a")
    with pytest.raises(
        recorder.RecordError,
        match=r"^LapicqueNeuron declares no parity word for lane go; "
        r"declare backends\.go\.parity before recording$",
    ):
        _record(recorder, tmp_path, "backend:go", _RUST_RUN)


def test_the_recorder_refuses_a_claim_scope_other_than_the_lane_s_word(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A caller cannot state a lane's claim scope; the descriptor states it."""
    recorder = _recorder_for(monkeypatch, "backend:python", _REFERENCE_RUN)
    with pytest.raises(
        recorder.RecordError,
        match=r"^claim scope 'exact' is not 'reference', "
        r"the word LapicqueNeuron gives lane python$",
    ):
        _record(recorder, tmp_path, "backend:python", _REFERENCE_RUN, claim_scope="exact")


def test_the_lane_toolchain_is_read_in_the_lane_s_build_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each lane's tools answer with their own version argument, in the folder that builds the lane."""
    monkeypatch.setenv("PATH", str(_stand_in_tools(tmp_path / "bin")))
    recorder = _load_recorder()
    root = recorder.REPO_ROOT.name
    assert recorder._lane_toolchain("rust") == {
        "rustc": "rustc -V in bridge",
        "cargo": "cargo -V in bridge",
    }
    assert recorder._lane_toolchain("rust_safety") == {
        "rustc": "rustc -V in rust",
        "cargo": "cargo -V in rust",
    }
    assert recorder._lane_toolchain("go") == {"go": "go version in go"}
    assert recorder._lane_toolchain("mojo") == {"mojo": "mojo --version in mojo"}
    assert recorder._lane_toolchain("julia") == {"julia": f"julia --version in {root}"}
    assert recorder._lane_toolchain("python") == {}
    assert recorder._lane_toolchain("") == {}


def test_a_lane_tool_that_fails_or_is_absent_is_left_out(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only a tool that answers is recorded; the others are absent, not empty."""
    monkeypatch.setenv("PATH", str(_stand_in_tools(tmp_path / "bin", failing=("cargo", "go"))))
    recorder = _load_recorder()
    assert recorder._lane_toolchain("rust") == {"rustc": "rustc -V in bridge"}
    assert recorder._lane_toolchain("go") == {}
    monkeypatch.setenv("PATH", str(tmp_path / "nowhere"))
    assert recorder._lane_toolchain("mojo") == {}

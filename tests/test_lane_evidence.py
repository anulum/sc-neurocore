# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — A run says which backend lane executed, and the report carries it

"""Bind the lane evidence a test run writes and a receipt can later read.

A model records, outside its own attributes, the lane its last complete run
executed and the native libraries that lane loaded. A test fixture writes that
record into the running test's JUnit report as two kinds of property, and the
report parser returns, per executed test case, its lane and library digests
together with every inconsistency. The lane is what the model ran, never what a
test asked for.
"""

from __future__ import annotations

import gc
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sc_neurocore.accel import lapicque as backends
from sc_neurocore.neurons import lane_runs
from sc_neurocore.neurons.lane_runs import LaneRun, lane_run, record_lane_run
from sc_neurocore.neurons.models.lapicque import LapicqueNeuron
from sc_neurocore.neurons.receipt_execution import (
    LANE_PROPERTY,
    NATIVE_BINARY_PROPERTY,
    CaseLane,
    junit_checks,
    junit_lane_evidence,
)
from tests.julia_requirement import require_julia

_REPO_ROOT = Path(__file__).resolve().parents[1]
_A = "a" * 64
_B = "b" * 64


def _report(*cases: str) -> str:
    """Return a JUnit report holding the given test case elements."""
    return f'<testsuite name="pytest">{"".join(cases)}</testsuite>'


def _case(name: str, *properties: tuple[str, str], skipped: bool = False) -> str:
    """Return one test case element of ``tests.test_x`` with the given properties."""
    inner = "".join(f'<property name="{key}" value="{value}" />' for key, value in properties)
    block = f"<properties>{inner}</properties>" if properties else ""
    skip = '<skipped message="skip" />' if skipped else ""
    return f'<testcase classname="tests.test_x" name="{name}">{block}{skip}</testcase>'


def _engine_library() -> Path:
    """Return the native module file that provides the Rust lane's entry point."""
    module = sys.modules[backends._EngineLapicqueCompleteFn.__module__]
    return Path(str(module.__file__))


def test_the_property_names_are_fixed() -> None:
    """Recorded reports carry these names; a renamed property would orphan every earlier report."""
    assert (LANE_PROPERTY, NATIVE_BINARY_PROPERTY) == (
        "sc_neurocore.lane",
        "sc_neurocore.native_binary",
    )


def test_a_report_with_lane_properties_gives_each_case_its_lane_and_libraries() -> None:
    """Every executed case is returned with its lane and the digests of its libraries."""
    report = _report(
        _case("test_rust", (LANE_PROPERTY, "rust"), (NATIVE_BINARY_PROPERTY, f"engine.so={_A}")),
        _case("test_python", (LANE_PROPERTY, "python")),
    )
    assert junit_lane_evidence(report) == (
        (
            CaseLane("tests.test_x.test_rust", "rust", (("engine.so", _A),)),
            CaseLane("tests.test_x.test_python", "python", ()),
        ),
        (),
    )


def test_a_case_without_a_lane_is_reported_and_not_returned() -> None:
    """A case whose run named no lane cannot stand for a lane."""
    report = _report(_case("test_a", (LANE_PROPERTY, "go")), _case("test_b"))
    cases, problems = junit_lane_evidence(report)
    assert [case.case for case in cases] == ["tests.test_x.test_a"]
    assert problems == ("test case tests.test_x.test_b reports no lane",)


def test_one_library_with_two_digests_in_one_report_is_reported() -> None:
    """Two runs of one lane that loaded different files under one name cannot both be its library."""
    report = _report(
        _case("test_a", (LANE_PROPERTY, "go"), (NATIVE_BINARY_PROPERTY, f"liblapicque.so={_A}")),
        _case("test_b", (LANE_PROPERTY, "go"), (NATIVE_BINARY_PROPERTY, f"liblapicque.so={_B}")),
    )
    _cases, problems = junit_lane_evidence(report)
    assert problems == (
        f"library liblapicque.so of lane go has two digests in one report: {_A}, {_B}",
    )


def test_a_conflict_seen_again_is_reported_once() -> None:
    """A third case that repeats the same two digests adds no second problem."""
    report = _report(
        _case("test_a", (LANE_PROPERTY, "go"), (NATIVE_BINARY_PROPERTY, f"liblapicque.so={_A}")),
        _case("test_b", (LANE_PROPERTY, "go"), (NATIVE_BINARY_PROPERTY, f"liblapicque.so={_B}")),
        _case("test_c", (LANE_PROPERTY, "go"), (NATIVE_BINARY_PROPERTY, f"liblapicque.so={_B}")),
    )
    _cases, problems = junit_lane_evidence(report)
    assert problems == (
        f"library liblapicque.so of lane go has two digests in one report: {_A}, {_B}",
    )


def test_two_lanes_may_name_their_libraries_alike() -> None:
    """The Go and the Mojo lane of one model each load a ``liblapicque.so``; one report holds both."""
    report = _report(
        _case("test_go", (LANE_PROPERTY, "go"), (NATIVE_BINARY_PROPERTY, f"liblapicque.so={_A}")),
        _case(
            "test_mojo", (LANE_PROPERTY, "mojo"), (NATIVE_BINARY_PROPERTY, f"liblapicque.so={_B}")
        ),
        _case(
            "test_go_again", (LANE_PROPERTY, "go"), (NATIVE_BINARY_PROPERTY, f"liblapicque.so={_A}")
        ),
    )
    assert junit_lane_evidence(report) == (
        (
            CaseLane("tests.test_x.test_go", "go", (("liblapicque.so", _A),)),
            CaseLane("tests.test_x.test_mojo", "mojo", (("liblapicque.so", _B),)),
            CaseLane("tests.test_x.test_go_again", "go", (("liblapicque.so", _A),)),
        ),
        (),
    )


def test_the_libraries_of_a_case_without_one_lane_are_not_compared() -> None:
    """A case that names no lane, or two, is no lane's evidence, so its library binds no lane's digest."""
    report = _report(
        _case("test_none", (NATIVE_BINARY_PROPERTY, f"liblapicque.so={_B}")),
        _case("test_go", (LANE_PROPERTY, "go"), (NATIVE_BINARY_PROPERTY, f"liblapicque.so={_A}")),
    )
    assert junit_lane_evidence(report) == (
        (CaseLane("tests.test_x.test_go", "go", (("liblapicque.so", _A),)),),
        ("test case tests.test_x.test_none reports no lane",),
    )


def test_a_case_that_names_two_lanes_is_reported() -> None:
    """One test case is the evidence of one lane."""
    report = _report(_case("test_a", (LANE_PROPERTY, "go"), (LANE_PROPERTY, "rust")))
    assert junit_lane_evidence(report) == (
        (),
        ("test case tests.test_x.test_a reports more than one lane: go, rust",),
    )


def test_a_lane_or_library_written_twice_counts_once() -> None:
    """A test that records the same run twice states one lane and one library."""
    report = _report(
        _case(
            "test_a",
            (LANE_PROPERTY, "mojo"),
            (NATIVE_BINARY_PROPERTY, f"liblapicque.so={_A}"),
            (LANE_PROPERTY, "mojo"),
            (NATIVE_BINARY_PROPERTY, f"liblapicque.so={_A}"),
        )
    )
    assert junit_lane_evidence(report) == (
        (CaseLane("tests.test_x.test_a", "mojo", (("liblapicque.so", _A),)),),
        (),
    )


@pytest.mark.parametrize("value", ["liblapicque.so", f"={_A}", "liblapicque.so=abc", f"x={_A}0"])
def test_a_malformed_library_property_is_reported(value: str) -> None:
    """A library property is a file name, an equals sign and a sha256 in hexadecimal.

    Parameters
    ----------
    value : str
        A property value that is not of that form.
    """
    report = _report(_case("test_a", (LANE_PROPERTY, "go"), (NATIVE_BINARY_PROPERTY, value)))
    cases, problems = junit_lane_evidence(report)
    assert problems == (f"test case tests.test_x.test_a reports a malformed library {value!r}",)
    assert cases == (CaseLane("tests.test_x.test_a", "go", ()),)


def test_another_property_of_a_case_is_ignored() -> None:
    """Properties other plugins write beside the lane evidence change nothing."""
    report = _report(_case("test_a", ("random_seed", "5"), (LANE_PROPERTY, "rust")))
    assert junit_lane_evidence(report) == ((CaseLane("tests.test_x.test_a", "rust", ()),), ())


def test_a_skipped_case_is_not_evidence_and_needs_no_lane() -> None:
    """Only executed cases are returned; a skipped one is neither returned nor reported."""
    report = _report(_case("test_a", (LANE_PROPERTY, "julia")), _case("test_b", skipped=True))
    assert junit_lane_evidence(report) == ((CaseLane("tests.test_x.test_a", "julia", ()),), ())


@pytest.mark.parametrize(
    "xml",
    [
        '<!DOCTYPE testsuite [<!ENTITY x "y">]><testsuite><testcase classname="a" name="b" />'
        "</testsuite>",
        "<testsuite><testcase",
        "<report />",
    ],
)
def test_a_report_the_checks_refuse_is_refused_here_too(xml: str) -> None:
    """A declared DTD, a broken document and a foreign root are refused as by junit_checks.

    Parameters
    ----------
    xml : str
        A report the execution checks refuse.
    """
    with pytest.raises(ValueError) as checks:
        junit_checks(xml)
    with pytest.raises(ValueError) as lanes:
        junit_lane_evidence(xml)
    assert str(lanes.value) == str(checks.value)


def test_a_case_without_its_identity_is_refused() -> None:
    """A test case without class name or name identifies no validator."""
    with pytest.raises(ValueError) as caught:
        junit_lane_evidence('<testsuite><testcase name="test_a" /></testsuite>')
    assert str(caught.value) == "JUnit testcase lacks its validator identity"


def test_lane_properties_do_not_change_the_counted_checks() -> None:
    """The execution checks count the same cases with and without lane properties."""
    bare = _report(_case("test_a"), _case("test_b", skipped=True))
    marked = _report(_case("test_a", (LANE_PROPERTY, "rust")), _case("test_b", skipped=True))
    assert junit_checks(marked) == junit_checks(bare)


def test_a_model_records_the_lane_of_its_last_complete_run() -> None:
    """No record before a run; the Python lane loads no library."""
    neuron = LapicqueNeuron()
    assert lane_run(neuron) is None
    neuron.simulate(20, 2.2, backend="python")
    assert lane_run(neuron) == LaneRun("python", ())


def test_the_rust_lane_names_the_engine_module_it_loaded() -> None:
    """The library of the Rust lane is the native module that holds its entry point."""
    assert backends._HAS_RUST
    neuron = LapicqueNeuron()
    neuron.simulate(20, 2.2, backend="rust")
    run = lane_run(neuron)
    assert run == LaneRun("rust", (_engine_library(),))
    assert run.libraries[0].is_file()


@pytest.mark.parametrize("backend", ["go", "mojo"])
def test_a_ctypes_lane_names_the_shared_library_it_loaded(backend: str) -> None:
    """The Go and Mojo lanes name the one shared library their loader opened.

    The repository's lane gate skips this case where the lane was never built
    and makes it an error where ``SC_NEUROCORE_REQUIRE_<LANE>=1`` is set.

    Parameters
    ----------
    backend : str
        A lane that loads a library through ctypes.
    """
    neuron = LapicqueNeuron()
    neuron.simulate(20, 2.2, backend=backend)
    run = lane_run(neuron)
    assert run is not None and run.lane == backend
    assert [path.name for path in run.libraries] == ["liblapicque.so"]
    assert run.libraries[0].parent.parts[-2:] == (
        ("neurons", "lapicque") if backend == "go" else ("mojo", "kernels")
    )


@pytest.mark.parametrize("backend", ["python", "rust", "julia", "go", "mojo"])
def test_the_lane_gate_asks_the_dispatch_s_own_loaders(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The availability predicate answers what the dispatch would find, lane by lane.

    Parameters
    ----------
    backend : str
        A lane of the model.
    monkeypatch : pytest.MonkeyPatch
        Makes every compiled lane unloadable.
    """
    for loader in ("ensure_julia_loaded", "ensure_go_loaded", "ensure_mojo_loaded"):
        monkeypatch.setattr(backends, loader, lambda: False)
    monkeypatch.setattr(backends, "_HAS_RUST", False)
    assert backends.backend_available(backend) is (backend == "python")
    monkeypatch.setattr(backends, "_HAS_RUST", True)
    for loader in ("ensure_julia_loaded", "ensure_go_loaded", "ensure_mojo_loaded"):
        monkeypatch.setattr(backends, loader, lambda: True)
    assert backends.backend_available(backend) is True
    assert backends.backend_available("cuda") is False


def test_the_reference_and_the_julia_lane_name_no_library_whatever_is_loaded() -> None:
    """Neither lane loads a native library of its own, also after the ctypes lanes were opened."""
    backends.ensure_go_loaded()
    backends.ensure_mojo_loaded()
    assert backends.lane_libraries("python") == ()
    assert backends.lane_libraries("julia") == ()
    assert backends.lane_libraries("cuda") == ()


def test_the_julia_lane_loads_no_native_library_of_its_own() -> None:
    """Julia runs the lane in its own runtime; the record names the lane and no library."""
    require_julia()
    assert backends.ensure_julia_loaded()
    neuron = LapicqueNeuron()
    neuron.simulate(20, 2.2, backend="julia")
    assert lane_run(neuron) == LaneRun("julia", ())


def test_a_run_that_raises_records_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The earlier record stays when a requested lane cannot run."""
    neuron = LapicqueNeuron()
    neuron.simulate(20, 2.2, backend="python")
    monkeypatch.setattr(backends, "ensure_go_loaded", lambda: False)
    with pytest.raises(RuntimeError):
        neuron.simulate(20, 2.2, backend="go")
    assert lane_run(neuron) == LaneRun("python", ())


def test_the_automatic_choice_records_the_lane_it_took(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no compiled lane available, the automatic choice runs and records the Python lane."""
    for loader in ("ensure_mojo_loaded", "ensure_julia_loaded", "ensure_go_loaded"):
        monkeypatch.setattr(backends, loader, lambda: False)
    monkeypatch.setattr(backends, "_HAS_RUST", False)
    neuron = LapicqueNeuron()
    neuron.simulate(20, 2.2)
    assert lane_run(neuron) == LaneRun("python", ())


def test_the_record_is_not_an_attribute_of_the_model() -> None:
    """Code that reads every attribute of a model sees the same names before and after a run."""
    neuron = LapicqueNeuron()
    before = set(vars(neuron))
    neuron.simulate(20, 2.2, backend="python")
    assert set(vars(neuron)) == before == set(vars(LapicqueNeuron()))


def test_a_later_run_replaces_the_record_of_the_same_model() -> None:
    """The record is the last complete run's; the registry keeps one entry per model."""
    assert backends._HAS_RUST
    neuron = LapicqueNeuron()
    neuron.simulate(5, 2.2, backend="python")
    held = len(lane_runs._RUNS)
    neuron.simulate(5, 2.2, backend="rust")
    assert lane_run(neuron) == LaneRun("rust", (_engine_library(),))
    assert len(lane_runs._RUNS) == held


def test_a_record_goes_with_its_model() -> None:
    """The registry holds no record of a model that no longer exists."""
    neuron = LapicqueNeuron()
    neuron.simulate(5, 2.2, backend="python")
    held = len(lane_runs._RUNS)
    del neuron
    gc.collect()
    assert len(lane_runs._RUNS) == held - 1


def test_a_model_that_cannot_be_referenced_weakly_is_refused() -> None:
    """A record that could outlive its model is not kept."""
    with pytest.raises(TypeError) as caught:
        record_lane_run(5, "python", ())
    assert str(caught.value) == (
        "a model whose lane is recorded must support weak references, got int"
    )


_INNER = '''
from sc_neurocore.neurons.models.lapicque import LapicqueNeuron
from tests.lane_evidence_support import lane_evidence  # the fixture under test


def test_rust(lane_evidence):
    neuron = LapicqueNeuron()
    neuron.simulate(20, 2.2, backend="rust")
    lane_evidence(neuron)


class FallingBack(LapicqueNeuron):
    """A model that silently runs the floor when the Rust lane is asked of it."""

    def simulate_complete(self, n_steps, drive=0.0, backend="auto"):
        return super().simulate_complete(n_steps, drive, "python" if backend == "rust" else backend)


def test_fallback(lane_evidence):
    neuron = FallingBack()
    neuron.simulate(20, 2.2, backend="rust")
    lane_evidence(neuron)


def test_no_run(lane_evidence):
    lane_evidence(LapicqueNeuron())
'''


@pytest.fixture(scope="module")
def inner_report(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, str]:
    """Run three real tests that use the fixture and return their JUnit report and output."""
    root = tmp_path_factory.mktemp("inner")
    (root / "test_inner.py").write_text(_INNER, encoding="utf-8")
    (root / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    report = root / "report.xml"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([str(_REPO_ROOT / "src"), str(_REPO_ROOT)])
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-c",
            str(root / "pytest.ini"),
            "--rootdir",
            str(root),
            f"--junitxml={report}",
            "-q",
            str(root / "test_inner.py"),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    return report.read_text(encoding="utf-8"), completed.stdout + completed.stderr


def test_the_fixture_writes_the_lane_and_the_library_digest(inner_report: tuple[str, str]) -> None:
    """A real run's report carries the lane and the digest of the library it loaded."""
    xml, _output = inner_report
    cases, _problems = junit_lane_evidence(xml)
    digest = hashlib.sha256(_engine_library().read_bytes()).hexdigest()
    assert CaseLane("test_inner.test_rust", "rust", ((_engine_library().name, digest),)) in cases


def test_the_report_names_the_lane_that_ran_not_the_one_asked_for(
    inner_report: tuple[str, str],
) -> None:
    """A model that falls back to the floor is reported on the floor, the lane that actually ran."""
    xml, _output = inner_report
    cases, _problems = junit_lane_evidence(xml)
    assert CaseLane("test_inner.test_fallback", "python", ()) in cases


def test_the_fixture_refuses_a_model_that_has_not_run(inner_report: tuple[str, str]) -> None:
    """Asking for the lane of a model that never ran fails the test and writes no lane."""
    xml, output = inner_report
    cases, problems = junit_lane_evidence(xml)
    assert "test_inner.test_no_run" not in [case.case for case in cases]
    assert problems == ("test case test_inner.test_no_run reports no lane",)
    assert "the model has no recorded lane run" in output
    assert junit_checks(xml)[0] == {
        "collected": 3,
        "passed": 2,
        "failed": 1,
        "errors": 0,
        "skipped": 0,
    }


def test_the_default_report_family_needs_no_warning(inner_report: tuple[str, str]) -> None:
    """The fixture writes properties without the warning record_property gives under xunit2."""
    _xml, output = inner_report
    assert "incompatible with junit_family" not in output

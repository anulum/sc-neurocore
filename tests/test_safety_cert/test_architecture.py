# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — safety-evidence architecture contracts

"""Pin responsibility ownership, acyclic imports, pickle paths, and mirror truth."""

from __future__ import annotations

import ast
import importlib
import pickle
from pathlib import Path
from types import ModuleType

from sc_neurocore.safety_cert import Requirement, SILLevel, SafetyStandard
from sc_neurocore.safety_cert import safety_cert as compatibility

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_ROOT = _REPO_ROOT / "src" / "sc_neurocore" / "safety_cert"
_MODULE_PREFIX = "sc_neurocore.safety_cert."
_OWNERSHIP = {
    "standards": {"ASILLevel", "SILLevel", "SafetyStandard"},
    "traceability": {"Requirement", "TraceabilityMatrix"},
    "failure_analysis": {"FMEDA", "FailureCategory", "FailureMode", "ReliabilityMetrics"},
    "formal_evidence": {
        "FormalProofCertificate",
        "FormalProperty",
        "FormalPropertyGapDetector",
        "ProofTestCoverage",
        "PropertyGap",
    },
    "timing_analysis": {"WCETAnalyzer", "WCETPath"},
    "compliance": {
        "ChecklistItem",
        "ComplianceChecklist",
        "CrossStandardMapper",
        "IEC62304Assessment",
        "SWClass",
    },
    "certification": {
        "CertificationGenerator",
        "CertificationPackage",
        "SafetyManualGenerator",
    },
    "fault_tolerance": {"CCFAnalysis", "CCFDefence", "HFTAssessment", "HFTLevel"},
    "change_impact": {"ChangeImpactTracker", "ChangeRecord"},
    "evidence": {"EvidenceBag", "EvidenceItem"},
}
_EXPECTED_IMPORTS = {
    "standards": set(),
    "traceability": {"standards"},
    "failure_analysis": {"standards"},
    "formal_evidence": {"standards"},
    "timing_analysis": set(),
    "compliance": {"standards"},
    "certification": {
        "compliance",
        "failure_analysis",
        "formal_evidence",
        "standards",
        "timing_analysis",
        "traceability",
    },
    "fault_tolerance": {"standards"},
    "change_impact": set(),
    "evidence": {"certification"},
}


def _module(name: str) -> ModuleType:
    return importlib.import_module(_MODULE_PREFIX + name)


def _internal_imports(name: str) -> set[str]:
    tree = ast.parse((_PACKAGE_ROOT / f"{name}.py").read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith(_MODULE_PREFIX):
                imports.add(node.module.removeprefix(_MODULE_PREFIX).split(".", maxsplit=1)[0])
    return imports


def test_each_public_class_has_one_responsibility_owner() -> None:
    """Every compatibility symbol must resolve to one domain module."""
    owned = set().union(*_OWNERSHIP.values())
    assert owned == set(compatibility.__all__)
    assert sum(len(symbols) for symbols in _OWNERSHIP.values()) == len(owned)
    for module_name, symbols in _OWNERSHIP.items():
        owner = _module(module_name)
        for symbol in symbols:
            assert getattr(owner, symbol) is getattr(compatibility, symbol)


def test_historical_pickle_module_paths_and_round_trip_are_stable() -> None:
    """Historical class locations must keep old serialized objects loadable."""
    for symbol in compatibility.__all__:
        public_class = getattr(compatibility, symbol)
        assert public_class.__module__ == compatibility.__name__
    requirement = Requirement(
        "REQ-ARCH-001",
        "Pin historical pickle resolution",
        SafetyStandard.IEC_61508,
        SILLevel.SIL_2,
        ["src/implementation.py"],
        ["tests/test_implementation.py"],
        "verified",
    )
    restored = pickle.loads(pickle.dumps(requirement))
    assert restored == requirement
    assert type(restored) is Requirement


def test_responsibility_import_graph_is_exact_and_acyclic() -> None:
    """Domain imports must match the reviewed one-way dependency graph."""
    observed = {name: _internal_imports(name) for name in _OWNERSHIP}
    assert observed == _EXPECTED_IMPORTS

    pending = {name: set(dependencies) for name, dependencies in observed.items()}
    resolved: set[str] = set()
    while pending:
        ready = {name for name, dependencies in pending.items() if dependencies <= resolved}
        assert ready, f"cycle in safety-evidence module graph: {pending}"
        resolved.update(ready)
        pending = {
            name: dependencies for name, dependencies in pending.items() if name not in ready
        }


def test_no_refactored_module_or_test_is_a_godfile() -> None:
    """The refactor must not move either GodFile into another oversized file."""
    assert len((_PACKAGE_ROOT / "safety_cert.py").read_text(encoding="utf-8").splitlines()) <= 120
    for module_name in _OWNERSHIP:
        line_count = len(
            (_PACKAGE_ROOT / f"{module_name}.py").read_text(encoding="utf-8").splitlines()
        )
        assert line_count <= 600, (module_name, line_count)
    for test_path in Path(__file__).parent.glob("test_*.py"):
        assert len(test_path.read_text(encoding="utf-8").splitlines()) <= 500, test_path.name


def test_false_certification_mirrors_are_absent_but_monitor_chain_remains() -> None:
    """Non-functional report mirrors must stay removed without monitor damage."""
    removed = (
        "src/sc_neurocore/accel/rust/safety/safety_cert.rs",
        "src/sc_neurocore/accel/go/services/safety_cert/safety_cert.go",
        "src/sc_neurocore/accel/go/services/safety_cert/__init__.py",
        "src/sc_neurocore/accel/julia/safety_cert/safety_cert.jl",
        "src/sc_neurocore/accel/mojo/kernels/safety_cert.mojo",
    )
    retained = (
        "src/sc_neurocore/accel/rust/safety/safety_monitor.rs",
        "src/sc_neurocore/accel/go/services/safety_monitor/safety_monitor.go",
        "src/sc_neurocore/accel/julia/safety_cert/safety_monitor.jl",
        "src/sc_neurocore/accel/mojo/kernels/safety_monitor.mojo",
    )
    assert all(not (_REPO_ROOT / relative).exists() for relative in removed)
    assert all((_REPO_ROOT / relative).is_file() for relative in retained)
    for registry in ("lib.rs", "mod.rs"):
        text = (
            _REPO_ROOT / "src" / "sc_neurocore" / "accel" / "rust" / "safety" / registry
        ).read_text(encoding="utf-8")
        assert "pub mod safety_cert;" not in text

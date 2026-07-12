# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


_HISTORICAL_LIF_REPORT = "docs/benchmarks/BENCHMARK_REPORT.md"
_ADC_KERNEL_REPORT = "benchmarks/results/bench_adc_to_spike_kernel.json"
_SPEEDUP_512_PATTERN = re.compile(
    r"\b512(?:\.\d+)?\s*[x×](?:[-\s\w]{0,48})?(?:speedup|faster|real-time)",
    re.IGNORECASE,
)
_SPEEDUP_512_ANCHORS = (
    _HISTORICAL_LIF_REPORT,
    _ADC_KERNEL_REPORT,
    "SC_NeuroCore_v3.6_WhitePaper_512x_Benchmarks.pdf",
    "512.4x",
    "525.51x",
    "653.28x",
)
_PACKAGE_BOUNDARY_DECISION_DOC = "docs/architecture/package_boundary_decision.md"
_PACKAGE_BOUNDARY_DECISIONS = (
    "core package",
    "optional extra",
    "contributor extra",
    "source-checkout research surface",
    "separate crate",
    "retired candidate",
)


def _repo_root() -> Path:
    """Return the repository root for path-stable public claim checks."""
    return Path(__file__).resolve().parents[1]


def _load_capability_manifest() -> ModuleType:
    """Load the generated-capability manifest tool from the live checkout."""
    tool_path = _repo_root() / "tools" / "capability_manifest.py"
    spec = importlib.util.spec_from_file_location("capability_manifest", tool_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _manifest_counts() -> dict[str, int]:
    """Return manifest inventory counts used in public documentation claims."""
    tool = _load_capability_manifest()
    manifest: dict[str, Any] = tool.build_capability_manifest(_repo_root())
    counts = manifest["counts"]
    return {
        "python_model_source_modules": int(counts["python_model_source_modules"]),
        "python_model_classes": int(counts["python_model_classes"]),
        "rust_pyo3_model_wrappers": int(counts["rust_pyo3_model_wrappers"]),
        "test_files": int(counts["test_files"]),
        "public_documentation_pages": int(counts["public_documentation_pages"]),
    }


def _rust_networkrunner_model_count() -> int:
    """Count Rust NetworkRunner models advertised by the live engine source."""
    source = (_repo_root() / "engine/src/network_runner.rs").read_text(encoding="utf-8")
    match = re.search(
        r"pub\s+fn\s+supported_models\s*\([^)]*\)\s*->\s*[^{]+\{(?P<body>.*?)\n\s*\}",
        source,
        re.DOTALL,
    )
    assert match is not None
    return len(re.findall(r'"[A-Za-z0-9_]+"', match.group("body")))


def _tracked_formal_paths(suffixes: tuple[str, ...]) -> list[Path]:
    """Return git-tracked ``hdl/formal`` files with one of ``suffixes``.

    The count is taken from the git index, not an on-disk ``rglob``, so the
    generated harnesses under the git-ignored ``hdl/formal/catalogue/*/``
    directories never inflate the public inventory: the number a contributor
    with a populated working tree sees must equal the number a clean CI
    checkout proves.
    """
    root = _repo_root()
    completed = subprocess.run(
        ["git", "ls-files", "-z", "--", "hdl/formal"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        root / relative for relative in completed.stdout.split("\0") if relative.endswith(suffixes)
    ]


def _formal_inventory() -> tuple[int, dict[str, int]]:
    """Count formal proof jobs and statement types from committed HDL sources."""
    statements = {"assert": 0, "assume": 0, "cover": 0}
    for path in sorted(
        _tracked_formal_paths((".v", ".sv", ".svh")),
        key=lambda candidate: candidate.as_posix(),
    ):
        text = path.read_text(encoding="utf-8")
        for keyword in statements:
            statements[keyword] += len(re.findall(rf"\b{keyword}\s*(?:property)?\s*\(", text))
    return len(_tracked_formal_paths((".sby",))), statements


def _compact_whitespace(text: str) -> str:
    """Collapse arbitrary whitespace for resilient prose comparisons."""
    return " ".join(text.split())


def _public_markdown_files() -> list[Path]:
    """Return tracked public Markdown claim surfaces for slogan checks."""
    root = _repo_root()
    candidates = [root / "README.md", root / "ROADMAP.md"]
    candidates.extend((root / "docs").rglob("*.md"))
    candidates.extend((root / "paper").rglob("*.md"))
    excluded_parts = {
        ".git",
        "docs/internal",
        "docs/_generated",
        "docs/assets",
        "docs/reports/generated",
    }
    public_files: list[Path] = []
    for path in sorted(candidates, key=lambda candidate: candidate.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if any(relative == part or relative.startswith(f"{part}/") for part in excluded_parts):
            continue
        public_files.append(path)
    return public_files


def _claim_window(text: str, start: int, end: int, *, radius: int = 320) -> str:
    """Return nearby prose around a matched claim for local anchor checks."""
    return text[max(0, start - radius) : min(len(text), end + radius)]


def _load_json_object(path: Path) -> dict[str, Any]:
    """Read a committed JSON artefact as an object for benchmark claim checks."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return cast(dict[str, Any], payload)


def _load_toml_object(path: Path) -> dict[str, Any]:
    """Read a committed TOML artefact as an object for metadata claim checks."""
    with path.open("rb") as config_file:
        payload = tomllib.load(config_file)
    assert isinstance(payload, dict)
    return payload


def _project_classifiers() -> tuple[str, ...]:
    """Return package classifiers from the live ``pyproject.toml`` metadata."""
    metadata = _load_toml_object(_repo_root() / "pyproject.toml")
    project = metadata["project"]
    assert isinstance(project, dict)
    raw_classifiers = project["classifiers"]
    assert isinstance(raw_classifiers, list)

    classifiers: list[str] = []
    for classifier in raw_classifiers:
        assert isinstance(classifier, str)
        classifiers.append(classifier)
    return tuple(classifiers)


def _project_optional_extra_names() -> tuple[str, ...]:
    """Return optional dependency group names from package metadata."""
    metadata = _load_toml_object(_repo_root() / "pyproject.toml")
    project = metadata["project"]
    assert isinstance(project, dict)
    optional_dependencies = project["optional-dependencies"]
    assert isinstance(optional_dependencies, dict)

    extra_names: list[str] = []
    for extra_name in optional_dependencies:
        assert isinstance(extra_name, str)
        extra_names.append(extra_name)
    return tuple(sorted(extra_names))


def _setuptools_excluded_package_roots() -> tuple[str, ...]:
    """Return top-level package roots excluded from wheel discovery."""
    metadata = _load_toml_object(_repo_root() / "pyproject.toml")
    tool = metadata["tool"]
    assert isinstance(tool, dict)
    setuptools = tool["setuptools"]
    assert isinstance(setuptools, dict)
    packages = setuptools["packages"]
    assert isinstance(packages, dict)
    finder = packages["find"]
    assert isinstance(finder, dict)
    raw_excludes = finder["exclude"]
    assert isinstance(raw_excludes, list)

    package_roots: set[str] = set()
    for raw_exclude in raw_excludes:
        assert isinstance(raw_exclude, str)
        package_root = raw_exclude.removesuffix(".*")
        if package_root.startswith("sc_neurocore."):
            package_roots.add(package_root)
    return tuple(sorted(package_roots))


def _cargo_workspace_excludes() -> tuple[str, ...]:
    """Return Cargo workspace excludes from the root manifest."""
    manifest = _load_toml_object(_repo_root() / "Cargo.toml")
    workspace = manifest["workspace"]
    assert isinstance(workspace, dict)
    raw_excludes = workspace["exclude"]
    assert isinstance(raw_excludes, list)

    excludes: list[str] = []
    for raw_exclude in raw_excludes:
        assert isinstance(raw_exclude, str)
        excludes.append(raw_exclude)
    return tuple(sorted(excludes))


def _assert_boundary_rows_cover_tokens(
    *,
    lines: tuple[str, ...],
    tokens: tuple[str, ...],
    token_kind: str,
) -> None:
    """Assert package-boundary decision rows classify every metadata token."""
    missing_tokens: list[str] = []
    for token in tokens:
        token_row = next((line for line in lines if f"`{token}`" in line), "")
        if not token_row or not any(
            decision in token_row for decision in _PACKAGE_BOUNDARY_DECISIONS
        ):
            missing_tokens.append(token)
    assert missing_tokens == [], f"missing {token_kind} package-boundary rows: {missing_tokens}"


def test_public_capability_snapshots_are_current() -> None:
    """Keep generated public capability snapshots in sync with repo inventory."""
    tool = _load_capability_manifest()

    tool.assert_outputs_current(_repo_root())


def test_current_public_entrypoints_use_generated_inventory_terms() -> None:
    """Ensure public entry points use generated inventory counts, not stale prose."""
    counts = _manifest_counts()
    rust_models = _rust_networkrunner_model_count()
    readme = (_repo_root() / "README.md").read_text(encoding="utf-8")
    docs_index = (_repo_root() / "docs/index.md").read_text(encoding="utf-8")
    neuron_reference = (_repo_root() / "docs/api/neuron_models.md").read_text(encoding="utf-8")
    mkdocs_metadata = (_repo_root() / "mkdocs.yml").read_text(encoding="utf-8")
    all_entrypoints = "\n".join((readme, docs_index, neuron_reference, mkdocs_metadata))

    assert f"{counts['python_model_source_modules']} Python model source modules" in all_entrypoints
    assert f"{counts['python_model_classes']} lazy-loaded Python model classes" in all_entrypoints
    assert f"{counts['rust_pyo3_model_wrappers']} Rust PyO3 model wrappers" in all_entrypoints
    assert f"{rust_models}-model NetworkRunner" in all_entrypoints
    assert f"| Python test files | {counts['test_files']} |" in readme

    stale_claims = (
        "174 neuron models",
        "174 Rust neuron models",
        "173 Neuron Models",
        "173 Rust implementations",
        "121 Python",
        "120**",
        "72 properties",
    )
    for stale_claim in stale_claims:
        assert stale_claim not in all_entrypoints


def test_public_formal_claims_match_hdl_inventory() -> None:
    """Ensure formal-method public claims match committed HDL inventory."""
    proof_jobs, statements = _formal_inventory()
    total = sum(statements.values())
    expected_summary = (
        f"{proof_jobs} SymbiYosys proof jobs and {total} formal statements "
        f"({statements['assert']} assert, {statements['assume']} assume, "
        f"{statements['cover']} cover)"
    )

    readme = (_repo_root() / "README.md").read_text(encoding="utf-8")
    comparison = (_repo_root() / "docs/benchmarks/comparison.md").read_text(encoding="utf-8")
    tutorial = (_repo_root() / "docs/tutorials/21_formal_verification.md").read_text(
        encoding="utf-8"
    )

    assert expected_summary in _compact_whitespace(readme)
    assert expected_summary in _compact_whitespace(comparison)
    assert expected_summary in _compact_whitespace(tutorial)


def test_rust_speedup_claims_are_artifact_anchored() -> None:
    """Ensure Rust speedup claims name the committed benchmark evidence."""
    readme = (_repo_root() / "README.md").read_text(encoding="utf-8")
    faq = (_repo_root() / "docs/guides/faq.md").read_text(encoding="utf-8")
    landscape = (_repo_root() / "docs/COMPETITIVE_LANDSCAPE.md").read_text(encoding="utf-8")

    speedup_evidence = "benchmarks/results/rust_scaling_benchmark.json"
    assert speedup_evidence in readme
    assert speedup_evidence in faq
    assert speedup_evidence in landscape
    assert "Brunel balanced-network" in readme
    assert "39-202x faster" not in readme
    assert "39–202× faster" not in readme


def test_512x_class_speedup_claims_are_artifact_anchored() -> None:
    """Preserve verified 512x-class evidence while requiring exact artefacts."""
    root = _repo_root()
    discrepancy = (
        root / "docs" / "reports" / "SC_NEUROCORE_DISCREPANCY_REMEDIATION_PLAN_2026-02-11.md"
    ).read_text(encoding="utf-8")
    historical_report = (root / _HISTORICAL_LIF_REPORT).read_text(encoding="utf-8")
    adc_report = _load_json_object(root / _ADC_KERNEL_REPORT)
    backends = cast(dict[str, dict[str, Any]], adc_report["backends"])

    assert _HISTORICAL_LIF_REPORT in discrepancy
    assert _ADC_KERNEL_REPORT in discrepancy
    assert "`512x`-class speedup evidence is real" in discrepancy
    assert "512.4x" in discrepancy
    assert "525.51x" in discrepancy
    assert "653.28x" in discrepancy
    assert "| LIF multi (100x100K) | 12911.296 | 25.196 | 512.4x | 400x |" in (historical_report)
    assert backends["rust"]["speedup_over_python"] == 525.51
    assert backends["mojo"]["speedup_over_python"] == 653.28


def test_adaptive_runtime_positioning_stays_bounded() -> None:
    """Keep v3.7 positioning tied to implemented runtime-substrate evidence."""
    study = (
        _repo_root() / "docs" / "research" / "SC_NEUROCORE_V3.7_ADAPTIVE_RUNTIME_ENGINE_STUDY.md"
    ).read_text(encoding="utf-8")
    compact = _compact_whitespace(study).lower()

    assert "shared runtime substrate" in compact
    assert "implemented v3.7 workloads" in compact
    assert "blanket claim" in compact
    assert "zero overhead polymorphism" not in compact


def test_conda_forge_claims_are_recipe_draft_until_published() -> None:
    """Keep conda-forge claims gated until upstream package publication exists."""
    root = _repo_root()
    public_claim_surfaces = [
        root / "ROADMAP.md",
        root / "docs" / "index.md",
        root / "docs" / "COMPETITIVE_LANDSCAPE.md",
        root / "docs" / "CHANGELOG.md",
        root / "conda" / "README.md",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in public_claim_surfaces)
    compact = _compact_whitespace(combined).lower()
    recipe = (root / "conda" / "meta.yaml").read_text(encoding="utf-8")

    assert "conda-forge recipe draft" in compact
    assert "not yet published on conda-forge" in compact
    assert "sha256: PLACEHOLDER" in recipe
    assert "ready for conda-forge distribution" not in compact
    assert "recipe ready for conda-forge distribution" not in compact
    assert "conda-forge recipe | **ready**" not in compact
    assert "conda install conda-forge::sc-neurocore" not in compact


def test_public_maturity_classifier_stays_beta_until_v4_release() -> None:
    """Keep package maturity below stable until public release evidence catches up."""
    root = _repo_root()
    classifiers = _project_classifiers()
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    bounded_surfaces = [
        root / "docs" / "index.md",
        root / "docs" / "architecture" / "architecture.md",
        root / "docs" / "architecture" / "AUTONOMOUS_LEARNING_ZENITH.md",
        root / "src" / "sc_neurocore" / "__init__.py",
        root / "src" / "sc_neurocore" / "bioware" / "bioware.py",
        root / "docs" / "API_REFERENCE.md",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in bounded_surfaces)
    compact = _compact_whitespace(combined).lower()

    assert "Development Status :: 4 - Beta" in classifiers
    assert "Development Status :: 5 - Production/Stable" not in classifiers
    assert "Development Status :: 5 - Production/Stable" not in pyproject
    assert "broad research surface" in compact
    assert "stable public api freeze" in compact
    assert "beta maturity" in compact
    assert "beta package surface" in compact
    assert "production-ready" not in compact
    assert "production ready" not in compact


def test_package_boundary_decision_covers_current_metadata() -> None:
    """Keep package, extras, and Rust workspace boundary decisions synchronized."""
    root = _repo_root()
    decision_path = root / _PACKAGE_BOUNDARY_DECISION_DOC
    decision = decision_path.read_text(encoding="utf-8")
    decision_lines = tuple(_compact_whitespace(line).lower() for line in decision.splitlines())
    decision_compact = _compact_whitespace(decision).lower()
    install_profiles = (root / "docs" / "guides" / "install_profiles.md").read_text(
        encoding="utf-8"
    )
    mkdocs_nav = (root / "mkdocs.yml").read_text(encoding="utf-8")

    assert "v4 package boundary decision" in decision_compact
    assert "base wheel stays the core package surface" in decision_compact
    assert "architecture/package_boundary_decision.md" in mkdocs_nav
    assert "../architecture/package_boundary_decision.md" in install_profiles
    _assert_boundary_rows_cover_tokens(
        lines=decision_lines,
        tokens=_project_optional_extra_names(),
        token_kind="optional-extra",
    )
    _assert_boundary_rows_cover_tokens(
        lines=decision_lines,
        tokens=_setuptools_excluded_package_roots(),
        token_kind="setuptools-exclude",
    )
    _assert_boundary_rows_cover_tokens(
        lines=decision_lines,
        tokens=_cargo_workspace_excludes(),
        token_kind="cargo-workspace-exclude",
    )
    assert "tbd" not in decision_compact
    assert "todo" not in decision_compact


def test_v4_product_boundary_names_core_and_research_extensions() -> None:
    """Keep the v4 product boundary explicit for core and research surfaces."""
    decision = (_repo_root() / _PACKAGE_BOUNDARY_DECISION_DOC).read_text(encoding="utf-8")
    decision_compact = _compact_whitespace(decision).lower()
    decision_lines = tuple(_compact_whitespace(line).lower() for line in decision.splitlines())
    core_surfaces = (
        "python package",
        "rust engine",
        "stochastic bitstreams",
        "verilog/rtl export",
        "nir",
    )
    research_surfaces = (
        "quantum",
        "bioware/mea",
        "photonic",
        "robotics",
        "world-model",
        "audio",
        "sleep",
        "swarm",
    )

    assert "v4 product boundary" in decision_compact
    assert "sc-neurocore-core" in decision_compact
    for surface in core_surfaces:
        assert surface in decision_compact
    assert "same evidence, coverage, docs, and packaging gates" in decision_compact

    missing_research_rows: list[str] = []
    for surface in research_surfaces:
        surface_row = next((line for line in decision_lines if f"`{surface}`" in line), "")
        if not surface_row or "research/experimental extension" not in surface_row:
            missing_research_rows.append(surface)
    assert missing_research_rows == [], (
        f"missing v4 research-extension classifications: {missing_research_rows}"
    )


def test_public_claim_language_excludes_unbounded_marketing_slogans() -> None:
    """Reject unbounded slogans and detached 512x-class speedup prose."""
    banned_literals = (
        "zero competitive gaps",
        "zero gaps",
    )
    offenders: list[str] = []
    for path in _public_markdown_files():
        relative = path.relative_to(_repo_root()).as_posix()
        text = path.read_text(encoding="utf-8")
        compact = _compact_whitespace(text).lower()
        for phrase in banned_literals:
            if phrase in compact:
                offenders.append(f"{relative}: banned phrase {phrase!r}")
        for match in _SPEEDUP_512_PATTERN.finditer(text):
            context = _claim_window(text, match.start(), match.end())
            if not any(anchor in context for anchor in _SPEEDUP_512_ANCHORS):
                offenders.append(f"{relative}: unanchored 512x-class speedup claim")

    assert offenders == []

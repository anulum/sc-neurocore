# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - security scanner CI packet docs test

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = REPO_ROOT / "docs" / "development" / "security_scanner_ci.md"


def _content_lower(path: Path) -> str:
    return path.read_text(encoding="utf-8").lower()


def test_security_scanner_ci_doc_exists() -> None:
    assert DOC_PATH.exists(), f"missing page: {DOC_PATH.relative_to(REPO_ROOT)}"


def test_security_scanner_ci_doc_contains_required_packet_phrases() -> None:
    text = _content_lower(DOC_PATH)

    required_phrases = [
        "current workflow generates",
        "security_scanner_manifest.json",
        "python/code plan",
        "rust/supply-chain plan",
        "model/data licence matrix copy",
        "model_data_license_matrix.json",
        "release security artifact index",
        "release sweep summary",
        "release_security_sweep.py",
        "semgrep summary",
        "run_semgrep_scanners.py",
        "hypothesis_fuzz_summary.json",
        "rust_proptest_summary.json",
        "security/release_artifacts_manifest.json",
        "mixed execution/planning envelope",
        "no direct `trivy fs`, `cargo-fuzz`, `gitleaks`, or similar",
    ]

    for phrase in required_phrases:
        assert phrase in text, f"missing required phrase: {phrase!r}"


def test_security_scanner_ci_doc_points_to_relevant_tools() -> None:
    text = _content_lower(DOC_PATH)

    for path_ref in (
        "tools/security_scanner_manifest.py",
        "tools/security_scan/ci_security_packet.py",
        "tools/security_scan/release_security_sweep.py",
        "tools/security_scan/run_semgrep_scanners.py",
        "tools/security_scan/python_code_scanner_plan.py",
        "tools/security_scan/rust_supply_chain_scanner_plan.py",
        "tools/security_scan/release_security_artifact_index.py",
    ):
        assert path_ref in text, f"missing tool reference: {path_ref!r}"


def test_security_scanner_ci_doc_has_no_forbidden_vendor_terms() -> None:
    text = _content_lower(DOC_PATH)
    forbidden = [
        "co" + "dex",
        "open" + "ai",
        "cla" + "ude",
        "anth" + "ropic",
        "gem" + "ini",
        "g" + "pt",
    ]

    for term in forbidden:
        assert term not in text, f"forbidden term in public doc: {term}"

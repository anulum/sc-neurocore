# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — GitHub App token format compatibility guard

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCAN_DIRS = (".github", "tools", "scripts", "src", "tests", "docs")
TEXT_SUFFIXES = {
    ".py",
    ".sh",
    ".yml",
    ".yaml",
    ".md",
    ".txt",
    ".toml",
    ".ini",
    ".cfg",
    ".json",
}
SKIP_SUBSTRINGS = (
    ".git/",
    "__pycache__/",
    ".mypy_cache/",
    ".pytest_cache/",
    "htmlcov/",
    "dist/",
    "build/",
    ".venv/",
    "docs/internal/",
)

# High-signal anti-patterns for legacy token assumptions.
RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"len\(\s*[A-Za-z0-9_]*token[A-Za-z0-9_]*\s*\)\s*([=!]=|<=|>=|<|>)\s*(40|41|42)\b"),
        "Token length guard around 40 chars may break with stateless installation tokens.",
    ),
    (
        re.compile(r"ghs_[A-Za-z0-9]{20,80}"),
        "Hardcoded ghs_ token literal/pattern without dot/underscore allowance may reject stateless format.",
    ),
    (
        re.compile(r"ghs_\[[^]]*A-Za-z0-9[^]]*\]\{20,80\}"),
        "Regex for ghs_ token appears length/pattern constrained and may reject stateless format.",
    ),
)


def _iter_files() -> list[Path]:
    files: list[Path] = []
    for rel in SCAN_DIRS:
        base = ROOT / rel
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            rel_posix = path.relative_to(ROOT).as_posix()
            if any(token in rel_posix for token in SKIP_SUBSTRINGS):
                continue
            if rel_posix == "tools/check_github_app_token_compat.py":
                continue
            files.append(path)
    return files


def _scan(path: Path) -> list[tuple[int, str, str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []
    findings: list[tuple[int, str, str]] = []
    for pattern, message in RULES:
        for match in pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            snippet = match.group(0)
            findings.append((line, message, snippet))
    return findings


def main() -> int:
    findings_total: list[tuple[str, int, str, str]] = []
    for file_path in _iter_files():
        for line, message, snippet in _scan(file_path):
            findings_total.append((file_path.relative_to(ROOT).as_posix(), line, message, snippet))

    if findings_total:
        print("GitHub App token compatibility check failed:\n")
        for rel_path, line, message, snippet in findings_total:
            print(f"- {rel_path}:{line}: {message}")
            print(f"    matched: {snippet}")
        print(
            "\nUse format-agnostic handling for installation tokens and avoid fixed length assumptions."
        )
        return 1

    print("GitHub App token compatibility check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
